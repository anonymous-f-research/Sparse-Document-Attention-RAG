from __future__ import annotations

from typing import List, Sequence, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from src.pipeline.utils import prompts
from src.pipeline.config import (
    DEVICE,
    LLM_BATCH_SIZE,
    MAX_GEN_TOKENS_document,
    MAX_GEN_TOKENS_false_answer,
    TEMPERATURE,
    TOP_P,
)


def load_llm(model_name: str, device: str = DEVICE):
    """
    Load a large language model and tokenizer for text generation.

    Args:
        model_name: HuggingFace model identifier (e.g., 'meta-llama/Llama-2-7b-chat-hf').
        device: Device to load the model on ('cpu', 'cuda', etc.). Defaults to config DEVICE.

    Returns:
        A tuple of (tokenizer, model) ready for generation tasks.

    Note:
        - Sets model to evaluation mode (no gradients)
        - Configures tokenizer padding: uses eos_token if pad_token is not defined
        - Sets padding_side to "left" for batch generation
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    model = model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer, model


def generate_batch_seq2seq(tokenizer, model, prompts_list, max_tokens=128, temperature=0.7, top_p=0.9):
    """
    Generate text sequences for a batch of prompts using a causal language model.

    Performs tokenization, padding, and generation with configurable sampling parameters.
    Automatically extracts only the generated (non-prompt) portion of output.

    Args:
        tokenizer: Tokenizer for encoding prompts and decoding outputs.
        model: Causal language model for generation (on appropriate device).
        prompts_list: List of prompt strings to generate from.
        max_tokens: Maximum number of new tokens to generate per prompt (default 128).
        temperature: Sampling temperature (default 0.7). Higher = more random, 0 = greedy.
        top_p: Nucleus sampling parameter (default 0.9). Controls cumulative probability mass.

    Returns:
        List of generated text strings (one per input prompt), with special tokens removed.

    Note:
        - Uses left-padding for batch compatibility
        - Automatically detects device from model parameters
        - Disables sampling if temperature == 0 (uses greedy decoding)
    """
    device = next(model.parameters()).device
    inputs = tokenizer(
        prompts_list,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(device)

    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_tokens,
            do_sample=True if temperature > 0 else False,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.eos_token_id,
        )

    decoded = []
    for i, out_ids in enumerate(outputs):
        gen_ids = out_ids[input_ids.shape[1]:]
        text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        decoded.append(text)
    return decoded


def generate_false_answers(tokenizer, model, queries: Sequence[str]) -> List[str]:
    """
    Generate plausible false answers for a batch of queries using an LLM.

    Uses a system prompt and chat template to instruct the model to create false but
    convincing answers. Processes queries in configurable batch sizes for efficiency.

    Args:
        tokenizer: Tokenizer with chat_template support for the model.
        model: Causal language model (e.g., Llama 2 Chat).
        queries: Sequence of query strings to generate false answers for.

    Returns:
        List of generated false answer strings (one per query).

    Note:
        - Uses SYSTEM_PROMPT_FALSE_ANSWER and USER_FALSE_ANSWER_PROMPT from config
        - Processes in batches of size LLM_BATCH_SIZE for memory efficiency
        - Generation parameters from config: MAX_GEN_TOKENS_false_answer, TEMPERATURE, TOP_P
    """
    prompts_list: List[str] = []
    for q in queries:
        user_content = prompts.USER_FALSE_ANSWER_PROMPT.format(query=q)
        chat_str = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": prompts.SYSTEM_PROMPT_FALSE_ANSWER},
                {"role": "user", "content": user_content},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        prompts_list.append(chat_str)

    outputs: List[str] = []
    for j in range(0, len(prompts_list), LLM_BATCH_SIZE):
        sub = prompts_list[j:j + LLM_BATCH_SIZE]
        out = generate_batch_seq2seq(
            tokenizer,
            model,
            sub,
            max_tokens=MAX_GEN_TOKENS_false_answer,
            temperature=TEMPERATURE,
            top_p=TOP_P,
        )
        outputs.extend(out)
    return outputs


def generate_malicious_docs(tokenizer, model, queries: Sequence[str], false_answers: Sequence[str]) -> List[str]:
    """
    Generate malicious documents containing false answers for injection into RAG systems.

    Creates realistic-looking documents that support false answers, designed to fool
    RAG systems into citing them as evidence. Uses both query and false answer context.

    Args:
        tokenizer: Tokenizer with chat_template support for the model.
        model: Causal language model for document generation.
        queries: Sequence of query strings (context for document generation).
        false_answers: Sequence of false answers to incorporate into documents.
                      Must have same length as queries.

    Returns:
        List of generated malicious document strings (one per query-answer pair).

    Note:
        - Uses SYSTEM_PROMPT_FALSE_DOC and USER_FALSE_DOC_PROMPT from config
        - Processes in batches of size LLM_BATCH_SIZE for efficiency
        - Documents are designed to be semantically plausible while supporting false answers
        - Generation parameters from config: MAX_GEN_TOKENS_document, TEMPERATURE, TOP_P
    """
    prompts_list: List[str] = []
    for q, fa in zip(queries, false_answers):
        user_content = prompts.USER_FALSE_DOC_PROMPT.format(query=q, false_answer=fa)
        chat_str = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": prompts.SYSTEM_PROMPT_FALSE_DOC},
                {"role": "user", "content": user_content},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        prompts_list.append(chat_str)

    outputs: List[str] = []
    for j in range(0, len(prompts_list), LLM_BATCH_SIZE):
        sub = prompts_list[j:j + LLM_BATCH_SIZE]
        out = generate_batch_seq2seq(
            tokenizer,
            model,
            sub,
            max_tokens=MAX_GEN_TOKENS_document,
            temperature=TEMPERATURE,
            top_p=TOP_P,
        )
        outputs.extend(out)
    return outputs


def build_attack_content_for_batch(
    preset_false_answer_groups,
    preset_malicious_doc_groups,
    need_attack_content: bool,
    tokenizer,
    model,
    queries: Sequence[str],
) -> Tuple[List[List[str]], List[List[str]]]:
    """
    Build attack content (false answers and malicious documents) for a batch of queries.

    Supports three modes:
    1. Use preset attack content if provided
    2. Return empty lists if attack content is not needed
    3. Generate new false answers and malicious documents from scratch

    Args:
        preset_false_answer_groups: Pre-generated false answers grouped by query (list-of-lists).
                                   If provided, generation is skipped.
        preset_malicious_doc_groups: Pre-generated malicious documents grouped by query (list-of-lists).
                                    If provided, generation is skipped.
        need_attack_content: Boolean flag indicating whether attack content should be generated.
        tokenizer: Tokenizer for LLM-based generation (if needed).
        model: Language model for generating content (if needed).
        queries: Sequence of query strings to generate attack content for.

    Returns:
        A tuple of (false_answer_groups, malicious_doc_groups) where:
            - false_answer_groups: List of lists, one group per query
            - malicious_doc_groups: List of lists, one group per query

        Each inner list contains the corresponding false answers or documents.
        Empty lists are returned for queries with no content.

    Note:
        Preset arguments take precedence over generation.
        If generation is needed, one false answer and one malicious doc are created per query.
    """
    if preset_false_answer_groups is not None and preset_malicious_doc_groups is not None:
        return preset_false_answer_groups, preset_malicious_doc_groups

    if not need_attack_content:
        return [[] for _ in queries], [[] for _ in queries]

    false_answers = generate_false_answers(tokenizer, model, queries)
    malicious_docs = generate_malicious_docs(tokenizer, model, queries, false_answers)

    false_groups: List[List[str]] = []
    mal_groups: List[List[str]] = []
    for fa, md in zip(false_answers, malicious_docs):
        false_groups.append([fa] if fa else [])
        mal_groups.append([md] if md else [])
    return false_groups, mal_groups
