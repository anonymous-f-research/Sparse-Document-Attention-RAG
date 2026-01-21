from __future__ import annotations

from typing import List, Optional

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from src.pipeline.config import MAX_GEN_TOKENS_RAG, TEMPERATURE
from src.pipeline.utils.prompts import SYSTEM_PROMPT_RAG, USER_RAG_PROMPT
from src.pipeline.utils.ranked_list import inject_malicious_docs_into_ranked_list


def compute_doc_knn_for_docs(ranker: SentenceTransformer, docs: List[str], k_neighbors: int) -> List[List[int]]:
    """
    Compute k-nearest neighbors for each document based on semantic similarity.

    Uses a SentenceTransformer to embed documents and find the k most similar documents
    for each one (excluding self-matches). Returns neighbor indices for use in document
    isolation masking during generation.

    Args:
        ranker: SentenceTransformer model for computing embeddings.
        docs: List of document texts to find neighbors for.
        k_neighbors: Number of nearest neighbors to find for each document.

    Returns:
        List of k-nearest neighbor lists (one per input document), where each inner list
        contains up to k document indices (0-indexed) of the most similar documents.
        Returns empty lists if k_neighbors <= 0 or input has <= 1 non-empty documents.

    Note:
        - Empty or whitespace documents are skipped in similarity computation
        - Neighbors are sorted by similarity (most similar first)
        - Self-matches are excluded from the neighbor list
        - If fewer than k neighbors exist, returns fewer than k results
    """
    n = len(docs)
    if k_neighbors <= 0 or n == 0:
        return [[] for _ in range(n)]

    nonempty = [(i, d) for i, d in enumerate(docs) if d and d.strip()]
    if len(nonempty) <= 1:
        return [[] for _ in range(n)]

    idxs, texts = zip(*nonempty)
    enc_inputs = ["passage: " + t for t in texts]
    emb = ranker.encode(enc_inputs, convert_to_tensor=True, normalize_embeddings=True).cpu().numpy().astype(np.float32)

    sims = np.matmul(emb, emb.T)
    neighbors_full: List[List[int]] = [[] for _ in range(n)]

    for row_idx, i_full in enumerate(idxs):
        row = sims[row_idx]
        order = np.argsort(-row)
        knn: List[int] = []
        for j in order:
            if j == row_idx:
                continue
            knn.append(int(idxs[j]))
            if len(knn) >= k_neighbors:
                break
        neighbors_full[i_full] = knn

    return neighbors_full


def build_blocked_causal_mask_full(
    base_seq_len: int,
    max_new_tokens: int,
    sys_user_len: int,
    doc_token_spans: list,
    qa_start: int,
    device: str = "cpu",
    doc_neighbors: Optional[List[List[int]]] = None,
):
    """
    Build a causal attention mask that isolates documents while allowing QA processing.

    Creates a mask for document-isolated generation where:
    - System/user input can attend to all prior tokens
    - Each document can only attend to system/user input and itself (plus optional neighbors)
    - QA section can attend to all prior tokens

    Args:
        base_seq_len: Length of the base prompt sequence (before generation).
        max_new_tokens: Maximum new tokens to generate (for forward compatibility).
        sys_user_len: Token length of system + user instruction section.
        doc_token_spans: List of (start_token, end_token) tuples for each document section.
        qa_start: Token position where the QA section begins.
        device: Device for mask tensor (default "cpu").
        doc_neighbors: Optional list of neighbor document indices for each document.
                      If provided, docs can also attend to their neighbors.

    Returns:
        Boolean mask tensor of shape (base_seq_len, base_seq_len) where mask[i, j] = True
        means token i can attend to token j (False means masked/blocked).

    Note:
        - Mask is built for causal (autoregressive) generation
        - Documents are isolated from each other unless they are neighbors
        - System/user section has standard causal masking
    """
    Lmax = base_seq_len
    mask = torch.zeros(Lmax, Lmax, dtype=torch.bool, device=device)

    for i in range(sys_user_len):
        mask[i, :i + 1] = True

    num_docs = len(doc_token_spans)
    use_neighbors = doc_neighbors is not None and len(doc_neighbors) == num_docs

    for d_idx, (d_start, d_end) in enumerate(doc_token_spans):
        for i in range(d_start, d_end):
            mask[i, :sys_user_len] = True
            mask[i, d_start:i + 1] = True
            if use_neighbors:
                for nbr in doc_neighbors[d_idx]:
                    if nbr < 0 or nbr >= num_docs:
                        continue
                    n_start, n_end = doc_token_spans[nbr]
                    mask[i, n_start:n_end] = True

    for i in range(qa_start, base_seq_len):
        mask[i, :i + 1] = True

    return mask


def generate_with_custom_mask(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    prompt_mask: torch.Tensor,
    max_new_tokens: int = 128,
):
    """
    Generate text with a custom attention mask for constrained decoding.

    Performs autoregressive generation where the initial prompt uses a custom mask,
    and subsequent tokens are generated with standard causal masking.

    Args:
        model: Causal language model with past_key_values support.
        tokenizer: Tokenizer for decoding (must have eos_token_id).
        input_ids: Token IDs of the input prompt (shape: [1, seq_len]).
        prompt_mask: Custom attention mask for the prompt (shape: [seq_len, seq_len]).
                    Boolean mask (True = attend, False = mask) or pre-computed attention matrix.
        max_new_tokens: Maximum number of tokens to generate (default 128).

    Returns:
        Generated text string (decoded output, special tokens removed).

    Note:
        - Stops early if EOS token is generated
        - First forward pass uses prompt_mask with KV cache
        - Subsequent passes use standard autoregressive masking
        - Respects model dtype for mask casting
        - Uses TEMPERATURE from config for sampling
    """
    device = input_ids.device
    L0 = input_ids.size(1)

    model_dtype = next(model.parameters()).dtype
    NEG_INF = torch.finfo(model_dtype).min

    if prompt_mask.dtype == torch.bool:
        attn = torch.zeros_like(prompt_mask, dtype=model_dtype, device=device)
        attn = attn.masked_fill(~prompt_mask.to(device), NEG_INF)
    else:
        attn = prompt_mask.to(device, dtype=model_dtype)

    attn = attn.unsqueeze(0).unsqueeze(1)

    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attn, use_cache=True)

    past_key_values = out.past_key_values
    generated = input_ids

    logits = out.logits[:, -1, :]
    if TEMPERATURE > 0:
        logits = logits / TEMPERATURE
        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
    else:
        next_token = torch.argmax(logits, dim=-1, keepdim=True)

    generated = torch.cat([generated, next_token], dim=1)

    for _ in range(max_new_tokens):
        with torch.no_grad():
            out = model(
                input_ids=generated[:, -1:],
                past_key_values=past_key_values,
                use_cache=True,
            )
        logits = out.logits[:, -1, :]
        past_key_values = out.past_key_values

        if TEMPERATURE > 0:
            logits = logits / TEMPERATURE
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        else:
            next_token = torch.argmax(logits, dim=-1, keepdim=True)

        generated = torch.cat([generated, next_token], dim=1)

        if next_token.item() == tokenizer.eos_token_id:
            break

    return tokenizer.decode(generated[0][L0:], skip_special_tokens=True).strip()


def build_rag_chat_and_spans(
    tokenizer,
    system_prompt: str,
    user_template: str,
    query: str,
    malicious_docs: List[str],
    retrieved_docs: list,
    add_attack_in_rank: int,
):
    """
    Build a RAG chat prompt with document isolation structure and token span annotations.

    Constructs a chat prompt containing the system message, user query with retrieved
    documents, and marks token positions for each document to enable selective attention
    masking during generation.

    Args:
        tokenizer: Tokenizer with chat_template support (e.g., Llama 2).
        system_prompt: System instruction text.
        user_template: User message template with {query} and {docs_text} placeholders.
        query: The original query string.
        malicious_docs: List of malicious documents to inject into ranking.
        retrieved_docs: List of documents from retrieval.
        add_attack_in_rank: Position parameter for injecting malicious docs.

    Returns:
        A tuple containing:
            - chat_str: Complete formatted chat string
            - sys_user_len: Token count for system + user instruction section
            - doc_token_spans: List of (start_tok, end_tok) tuples for each document
            - qa_start: Token position where QA section begins
            - ranked_docs: Final ranked document list (with malicious docs injected)

    Note:
        - Documents are formatted as numbered bullet points
        - Token positions are computed by tokenizing substrings
        - Malicious docs are injected according to add_attack_in_rank parameter
    """
    ranked_docs = inject_malicious_docs_into_ranked_list(
        base_docs=retrieved_docs,
        malicious_docs=malicious_docs,
        attack_pos=add_attack_in_rank,
    )

    numbered_docs = [f"- {d.strip()}" for d in ranked_docs if d and d.strip()]
    docs_text = "\n\n".join(numbered_docs)

    user_content = user_template.format(query=query, docs_text=docs_text)

    chat_str = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )

    doc_texts = numbered_docs
    doc_text_positions = []
    search_from = 0
    for dt in doc_texts:
        pos = chat_str.find(dt, search_from)
        if pos == -1:
            pos = search_from
        doc_text_positions.append(pos)
        search_from = pos + len(dt)

    question_marker = "- Question:"
    q_pos = chat_str.find(question_marker)
    if q_pos == -1:
        q_pos = len(chat_str)

    first_doc_pos = doc_text_positions[0] if doc_text_positions else q_pos
    sys_user_substr = chat_str[:first_doc_pos]
    sys_user_len = len(tokenizer(sys_user_substr, return_tensors="pt").input_ids[0])

    doc_token_spans = []
    for dt, start_char in zip(doc_texts, doc_text_positions):
        doc_substr_before = chat_str[:start_char]
        doc_start_tok = len(tokenizer(doc_substr_before, return_tensors="pt").input_ids[0])
        doc_full_substr = chat_str[: start_char + len(dt)]
        doc_end_tok = len(tokenizer(doc_full_substr, return_tensors="pt").input_ids[0])
        doc_token_spans.append((doc_start_tok, doc_end_tok))

    qa_substr = chat_str[:q_pos]
    qa_start = len(tokenizer(qa_substr, return_tensors="pt").input_ids[0])

    return chat_str, sys_user_len, doc_token_spans, qa_start, ranked_docs


def run_rag_with_doc_isolation(
    model,
    tokenizer,
    query: str,
    malicious_docs: List[str],
    retrieved_docs: list,
    max_new_tokens: int,
    device: str,
    add_attack_in_rank: int,
    ranker: SentenceTransformer,
    doc_neighbors_k: int,
):
    """
    Generate RAG answers with document isolation defense enabled.

    Executes constrained autoregressive generation where documents are isolated from
    each other during attention, preventing the model from mixing information across
    documents and reducing susceptibility to adversarial injections.

    Args:
        model: Causal language model for RAG answer generation.
        tokenizer: Tokenizer with chat template support.
        query: The original query string.
        malicious_docs: Injected adversarial documents.
        retrieved_docs: Documents retrieved by the RAG system.
        max_new_tokens: Maximum tokens in generated answer.
        device: Device for model computation (cuda/cpu).
        add_attack_in_rank: Position parameter for document injection.
        ranker: SentenceTransformer for computing document neighbors.
        doc_neighbors_k: Number of document neighbors to consider (0 = full isolation).

    Returns:
        Generated answer string with document isolation applied.

    Note:
        - Documents can attend to k-nearest neighbors if doc_neighbors_k > 0
        - doc_neighbors_k = 0 results in strict document isolation
        - Uses SYSTEM_PROMPT_RAG and USER_RAG_PROMPT from prompts module
    """
    chat_str, sys_user_len, doc_token_spans, qa_start, ranked_docs = build_rag_chat_and_spans(
        tokenizer=tokenizer,
        system_prompt=SYSTEM_PROMPT_RAG,
        user_template=USER_RAG_PROMPT,
        query=query,
        malicious_docs=malicious_docs,
        retrieved_docs=retrieved_docs,
        add_attack_in_rank=add_attack_in_rank,
    )

    if doc_neighbors_k and doc_neighbors_k > 0:
        doc_neighbors = compute_doc_knn_for_docs(ranker=ranker, docs=ranked_docs, k_neighbors=doc_neighbors_k)
    else:
        doc_neighbors = None

    encoded = tokenizer(chat_str, return_tensors="pt").to(device)
    input_ids = encoded["input_ids"]
    seq_len = input_ids.size(1)

    base_mask = build_blocked_causal_mask_full(
        base_seq_len=seq_len,
        max_new_tokens=MAX_GEN_TOKENS_RAG,
        sys_user_len=sys_user_len,
        doc_token_spans=doc_token_spans,
        qa_start=qa_start,
        device=device,
        doc_neighbors=doc_neighbors,
    )

    return generate_with_custom_mask(model, tokenizer, input_ids, base_mask, max_new_tokens=max_new_tokens)
