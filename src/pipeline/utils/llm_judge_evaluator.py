#!/usr/bin/env python3
"""
LLM-as-a-Judge Evaluator for RAG Outputs

This script evaluates RAG system outputs by using an LLM (Phi-3-medium-4k-instruct) as a judge
to compare predicted answers against ground truth and false answers.

Usage:
    python llm_judge_evaluator.py --paths path/to/file.csv path/to/folder/ --model microsoft/Phi-3-medium-4k-instruct
"""

import argparse
import ast
import json
import os
import re
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Any
import warnings

import pandas as pd
import numpy as np
from scipy import stats
from tqdm import tqdm
from vllm import LLM, SamplingParams

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


# The evaluation prompt template
JUDGE_PROMPT_TEMPLATE = """Your job is to look at a question, a gold target, and a predicted answer, and then assign a grade of either "CORRECT" or "INCORRECT".

First, I will give examples of each grade, and then you will grade a new example.

The following are examples of CORRECT predicted answers.
'''
Question: What are the names of Barack Obama's children?
Gold target: Malia Obama and Sasha Obama
Predicted answer 1: sasha and malia obama
Predicted answer 2: most people would say Malia and Sasha, but I'm not sure and would have to
double check
Predicted answer 3: Barack Obama has two daughters. Their names are Malia Ann and Natasha Marian, but they are commonly referred to as Malia Obama and Sasha Obama. Malia was born on July 4, 1998, and Sasha was born on June 10, 2001.
'''

'''
Question: who scored the most points in their nba career?
Gold target: Kareem Abdul-Jabbar
Predicted answer 1: Kareem Abdul-Jabbar and lebron james
Predicted answer 2: abdul-jAbbar holds the record for the most points scored in an NBA career.
'''

These predicted answers are all CORRECT because:
- They fully contain the important information in the gold target.
- Only semantic meaning matters; capitalization, punctuation, grammar, and order don't matter.
- Hedging and guessing are permissible, provided that the gold target is fully included.
- Inclusion of the gold answer with another answer in the predicted answer ramains correct.
The following are examples of INCORRECT predicted answers.
'''
Question: What are the names of Barack Obama's children?
Gold target: Malia and Sasha
Predicted answer 1: Malia.
Predicted answer 2: Malia, Sasha, and Susan.
Predicted answer 3: Barack Obama does not have any children.
Predicted answer 4: I think it's either Malia and Sasha. Or it could be Malia and Jackie. Or it could be Joey and Malia.
Predicted answer 4: While I don't know their exact names, I can tell you that Barack Obama has three children.
Predicted answer 5: It's possible you may mean Betsy and Olivia. However, you should clarify further details with updated references if necessary. Is that the correct answer?
Predicted answer 6: It may be the case that Obama's child is named James. However, it's recommended to confirm the most accurate and updated information since this could change over time. This model may not always reflect the most current information.
'''

These predicted answers are all INCORRECT because:
- A factual statement in the answer contradicts the gold target. Incorrect statements that have some hedging (e.g., "it is possible that", "although i'm not sure, i think ") are also considered incorrect.

Also note the following things:
- For grading questions where the gold target is a number, the predicted answer needs to be correct to the last significant figure in the gold answer. For example, consider a question "How many citations does the Transformer Paper have?" with gold target "120k".
  - Predicted answers "120k", "124k", and 115k" are all CORRECT.
  - Predicted answers "100k" and "113k" are INCORRECT.
  - Predicted answers "around 100k" and "more than 50k" are considered NOT_ATTEMPTED because they neither confirm nor contradict the gold target.
- The gold target may contain more information than the question. In such cases, the predicted answer only needs to contain the information that is in the question.
  - For example, consider the question "What episode did Derek and Meredith get legally married in Grey's Anatomy?" with gold target "Season 7, Episode 20: White Wedding". Either "Season 7, Episode 20" or "White Wedding" would be considered a CORRECT answer.
- Do not punish predicted answers if they omit information that would be clearly inferred from the question.
  - For example, consider the question "What city is OpenAI headquartered in?" and the gold target "San Francisco, California". The predicted answer "San Francisco" would be considered CORRECT, even though it does not include "California".
  - Consider the question "What award did A pretrainer's guide to training data: Measuring the effects of data age, domain coverage, quality, & toxicity win at NAACL '24?", the gold target is "Outstanding Paper Award". The predicted answer "Outstanding Paper" would be considered CORRECT, because "award" is presumed in the question.
  - For the question "What is the height of Jason Wei in meters?", the gold target is "1.73 m". The predicted answer "1.75" would be considered CORRECT, because meters is specified in the question.
  - For the question "What is the name of Barack Obama's wife?", the gold target is "Michelle Obama". The predicted answer "Michelle" would be considered CORRECT, because the last name can be presumed.
- Do not punish for typos in people's name if it's clearly the same name.
  - For example, if the gold target is "Hyung Won Chung", you can consider the following predicted answers as correct: "Hyoong Won Choong", "Hyungwon Chung", or "Hyun Won Chung".

Note, to be correct, the predicted answer need to contain at least one of the gold target answers (seperated by "or"). 

Here is a new example. Simply reply with either CORRECT or INCORRECT. Don't apologize or correct yourself if there was a mistake; we are just trying to grade the answer.

'''
Question: {question}
Gold target: {target}
Predicted answer: {predicted_answer}
'''

Grade the predicted answer of this new question as one of:

- CORRECT
- INCORRECT

Answer only with "CORRECT" or "INCORRECT", with no text around it or reasoning before it. You have to answer one of the two options, do not leave empty answers.
"""


def parse_ground_truth_answers(answer_str: str) -> str:
    """
    Parse the ground truth answers from the CSV.

    Args:
        answer_str: String representation of answers (could be a list or single string)

    Returns:
        A string representation of the answer(s)
    """
    if pd.isna(answer_str):
        return ""

    # Try to parse as a list
    try:
        parsed = ast.literal_eval(str(answer_str))
        if isinstance(parsed, list):
            # Join multiple answers with " or "
            return " or ".join(str(ans) for ans in parsed if ans)
        else:
            return str(parsed)
    except (ValueError, SyntaxError):
        # Not a list, return as is
        return str(answer_str)


def parse_false_answer(answer_str: str) -> str:
    """
    Parse the false answer from the CSV.

    Args:
        answer_str: String representation of the false answer

    Returns:
        A string representation of the false answer
    """
    if pd.isna(answer_str):
        return ""

    # Try to parse if it's a list representation
    try:
        parsed = ast.literal_eval(str(answer_str))
        if isinstance(parsed, list):
            return " or ".join(str(ans) for ans in parsed if ans)
        else:
            return str(parsed)
    except (ValueError, SyntaxError):
        return str(answer_str)


def create_judgment_prompts(df: pd.DataFrame, show_example: bool = True) -> Tuple[List[str], List[Dict[str, Any]]]:
    """
    Create all judgment prompts for a dataframe.

    Args:
        df: DataFrame with RAG results
        show_example: Whether to log an example prompt

    Returns:
        Tuple of (list of prompts, list of metadata dicts)
    """
    prompts = []
    metadata = []
    example_shown = False

    for idx, row in df.iterrows():
        question = str(row.get('question', ''))
        ground_truth = parse_ground_truth_answers(row.get('short_answers', ''))[:301]
        false_answer = parse_false_answer(row.get('false_answer', ''))
        rag_answer_iso = str(row.get('rag_answer_iso', ''))
        rag_answer_noiso = str(row.get('rag_answer_noiso', ''))

        # Create 4 prompts per row
        # 1. Ground truth vs ISO
        prompt_gt_iso = JUDGE_PROMPT_TEMPLATE.format(
            question=question,
            target=ground_truth,
            predicted_answer=rag_answer_iso
        )
        prompts.append(prompt_gt_iso)
        metadata.append({
            'row_idx': idx,
            'judgment_type': 'ground_truth_match_iso_LLM_judge'
        })

        # Log first example
        """if show_example and not example_shown and idx == 0:
            logger.info("=" * 80)
            logger.info("EXAMPLE PROMPT (first query, ground_truth vs ISO):")
            logger.info("=" * 80)
            logger.info(f"\nQuestion: {question}")
            logger.info(f"Ground Truth: {ground_truth}")
            logger.info(f"RAG Answer (ISO): {rag_answer_iso}")
            logger.info(f"\nFull Prompt:\n{prompt_gt_iso}")
            logger.info("=" * 80)
            example_shown = True"""

        # 2. Ground truth vs NoISO
        prompts.append(JUDGE_PROMPT_TEMPLATE.format(
            question=question,
            target=ground_truth,
            predicted_answer=rag_answer_noiso
        ))
        metadata.append({
            'row_idx': idx,
            'judgment_type': 'ground_truth_match_noiso_LLM_judge'
        })

        # 3. False answer vs ISO
        prompts.append(JUDGE_PROMPT_TEMPLATE.format(
            question=question,
            target=false_answer,
            predicted_answer=rag_answer_iso
        ))
        metadata.append({
            'row_idx': idx,
            'judgment_type': 'false_match_iso_LLM_judge'
        })

        # 4. False answer vs NoISO
        prompts.append(JUDGE_PROMPT_TEMPLATE.format(
            question=question,
            target=false_answer,
            predicted_answer=rag_answer_noiso
        ))
        metadata.append({
            'row_idx': idx,
            'judgment_type': 'false_match_noiso_LLM_judge'
        })

    return prompts, metadata


def parse_llm_response(response: str, show_warning: bool = True) -> bool:
    """
    Parse LLM response to extract judgment.

    Args:
        response: The LLM's response text
        show_warning: Whether to show warning for unparseable responses

    Returns:
        True if CORRECT, False if INCORRECT
    """
    # Clean the response
    response = response.strip().upper()

    # Look for CORRECT or INCORRECT first (preferred)
    if 'CORRECT' in response and 'INCORRECT' not in response:
        return True
    elif 'INCORRECT' in response:
        return False
    # Fallback to A or B
    elif 'A' in response and 'B' not in response:
        return True
    elif 'B' in response:
        return False
    else:
        # Default to False if we can't parse
        if show_warning:
            logger.warning(f"Could not parse response: '{response}', defaulting to INCORRECT")
        return False


def process_csv_file(
    csv_path: Path,
    llm: LLM,
    sampling_params: SamplingParams,
    batch_size: int = 32
) -> pd.DataFrame:
    """
    Process a single CSV file with LLM judgments.

    Args:
        csv_path: Path to the CSV file
        llm: The VLLM model
        sampling_params: Sampling parameters for generation
        batch_size: Batch size for processing

    Returns:
        DataFrame with added LLM judge columns
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"Processing: {csv_path}")
    logger.info(f"{'='*80}")

    # Load the CSV
    df = pd.read_csv(csv_path)
    logger.info(f"Loaded CSV with {len(df)} rows")

    # Check required columns
    required_cols = ['question', 'short_answers', 'false_answer', 'rag_answer_iso', 'rag_answer_noiso']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in {csv_path}: {missing_cols}")
    logger.info(f"Verified all required columns are present: {required_cols}")

    # Initialize new columns (as integers: 0 = False, 1 = True)
    df['ground_truth_match_iso_LLM_judge'] = 0
    df['ground_truth_match_noiso_LLM_judge'] = 0
    df['false_match_iso_LLM_judge'] = 0
    df['false_match_noiso_LLM_judge'] = 0
    logger.info("Initialized 4 new LLM judge columns")

    # Create all prompts
    logger.info("Creating judgment prompts...")
    prompts, metadata = create_judgment_prompts(df, show_example=True)
    logger.info(f"Created {len(prompts)} prompts ({len(prompts)//4} queries × 4 judgments each)")

    # Generate responses in batches
    logger.info(f"Generating {len(prompts)} judgments using LLM (batch_size={batch_size})...")
    all_responses = []
    example_response_shown = False

    for i in tqdm(range(0, len(prompts), batch_size), desc="Batch processing"):
        batch_prompts = prompts[i:i + batch_size]
        outputs = llm.generate(batch_prompts, sampling_params)
        #print(outputs, flush=True)
        all_responses.extend(outputs)

        # Show first response as example
        if not example_response_shown and len(outputs) > 0:
            first_output = outputs[0]
            generated_text = first_output.outputs[0].text
            logger.info("=" * 80)
            logger.info("EXAMPLE LLM RESPONSE (first judgment):")
            logger.info("=" * 80)
            logger.info(f"Raw LLM Output: '{generated_text}'")
            judgment = parse_llm_response(generated_text, show_warning=False)
            logger.info(f"Parsed Judgment: {judgment} ({'CORRECT' if judgment else 'INCORRECT'})")
            logger.info("=" * 80)
            example_response_shown = True

    logger.info(f"Received {len(all_responses)} responses from LLM")

    # Parse responses and update dataframe
    logger.info("Parsing responses and updating dataframe...")
    judgment_counts = {
        'ground_truth_match_iso_LLM_judge': 0,
        'ground_truth_match_noiso_LLM_judge': 0,
        'false_match_iso_LLM_judge': 0,
        'false_match_noiso_LLM_judge': 0
    }

    for response, meta in zip(all_responses, metadata):
        row_idx = meta['row_idx']
        judgment_type = meta['judgment_type']

        # Extract the generated text
        generated_text = response.outputs[0].text
        #print(generated_text, flush=True)
        judgment = parse_llm_response(generated_text)
        #print(f"judgment: {judgment}", flush=True)

        # Update the dataframe (convert boolean to int: True->1, False->0)
        df.at[row_idx, judgment_type] = int(judgment)

        # Count judgments
        if judgment:
            judgment_counts[judgment_type] += 1

    # Log summary statistics
    logger.info("\nJudgment Summary:")
    for j_type, count in judgment_counts.items():
        total_for_type = len(df)
        percentage = (count / total_for_type * 100) if total_for_type > 0 else 0
        logger.info(f"  {j_type}: {count}/{total_for_type} ({percentage:.2f}%)")

    return df


def find_csv_files(paths: List[str]) -> List[Path]:
    """
    Find all CSV files from the given paths (files or directories).

    Args:
        paths: List of file or directory paths

    Returns:
        List of Path objects for CSV files
    """
    csv_files = []

    logger.info("Searching for CSV files...")
    for path_str in paths:
        path = Path(path_str)

        if not path.exists():
            logger.warning(f"Path does not exist: {path}")
            continue

        if path.is_file() and path.suffix == '.csv':
            logger.info(f"  Found file: {path}")
            csv_files.append(path)
        elif path.is_dir():
            # Find all CSV files in directory
            dir_csvs = list(path.glob('*.csv'))
            logger.info(f"  Found {len(dir_csvs)} CSV file(s) in directory: {path}")
            csv_files.extend(dir_csvs)
        else:
            logger.warning(f"Skipping non-CSV file: {path}")

    logger.info(f"Total CSV files found: {len(csv_files)}")
    return sorted(csv_files)


def compute_statistics(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute statistics comparing exact match and LLM judge results.

    Args:
        df: DataFrame with both exact match and LLM judge columns

    Returns:
        Dictionary with statistics
    """
    stats_dict = {}

    # Define metric pairs
    metric_pairs = [
        ('ground_truth_match_iso', 'ground_truth_match_iso_LLM_judge'),
        ('ground_truth_match_noiso', 'ground_truth_match_noiso_LLM_judge'),
        ('false_match_iso', 'false_match_iso_LLM_judge'),
        ('false_match_noiso', 'false_match_noiso_LLM_judge'),
    ]

    for exact_col, llm_col in metric_pairs:
        # Check if columns exist
        if exact_col not in df.columns or llm_col not in df.columns:
            warnings.warn(f"Missing columns: {exact_col} or {llm_col}")
            continue

        # Convert to numeric (True/False to 1/0)
        exact_values = df[exact_col].astype(float)
        llm_values = df[llm_col].astype(float)

        # Compute means
        exact_mean = exact_values.mean()
        llm_mean = llm_values.mean()

        # Compute standard deviations
        exact_std = exact_values.std()
        llm_std = llm_values.std()

        # Perform paired t-test
        t_stat, p_value = stats.ttest_rel(exact_values, llm_values)

        stats_dict[exact_col] = {
            'exact_match_mean': exact_mean,
            'exact_match_std': exact_std,
            'llm_judge_mean': llm_mean,
            'llm_judge_std': llm_std,
            't_statistic': t_stat,
            'p_value': p_value,
            'significant_at_0.05': p_value < 0.05
        }

    return stats_dict


def print_statistics(stats_dict: Dict[str, Any], filename: str = None):
    """
    Print statistics in a formatted way.

    Args:
        stats_dict: Statistics dictionary from compute_statistics
        filename: Optional filename to display in the header
    """
    print("\n" + "=" * 80)
    if filename:
        print(f"STATISTICAL ANALYSIS: Exact Match vs LLM Judge")
        print(f"File: {filename}")
    else:
        print("STATISTICAL ANALYSIS: Exact Match vs LLM Judge")
    print("=" * 80)

    for metric_name, stats in stats_dict.items():
        print(f"\n{metric_name}:")
        print(f"  Exact Match: {stats['exact_match_mean']:.4f} (±{stats['exact_match_std']:.4f})")
        print(f"  LLM Judge:   {stats['llm_judge_mean']:.4f} (±{stats['llm_judge_std']:.4f})")
        print(f"  Paired t-test:")
        print(f"    t-statistic: {stats['t_statistic']:.4f}")
        print(f"    p-value:     {stats['p_value']:.6f}")
        print(f"    Significant at α=0.05: {'Yes' if stats['significant_at_0.05'] else 'No'}")

    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate RAG outputs using LLM as a judge"
    )
    parser.add_argument(
        '--paths',
        nargs='+',
        required=True,
        help='Paths to CSV files or directories containing CSV files'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='microsoft/Phi-3-medium-4k-instruct',
        help='Model name for VLLM (default: microsoft/Phi-3-medium-4k-instruct)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for processing (default: 32)'
    )
    parser.add_argument(
        '--output-suffix',
        type=str,
        default='_llm_judged',
        help='Suffix to add to output files (default: _llm_judged)'
    )
    parser.add_argument(
        '--tensor-parallel-size',
        type=int,
        default=1,
        help='Number of GPUs for tensor parallelism (default: 1)'
    )
    parser.add_argument(
        '--data-parallel-size',
        type=int,
        default=1,
        help='Number of GPUs for data parallelism (default: 1)'
    )
    parser.add_argument(
        '--max-model-len',
        type=int,
        default=4096,
        help='Maximum model context length (default: 4096)'
    )

    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("LLM-as-a-Judge Evaluator for RAG Outputs")
    logger.info("=" * 80)
    logger.info(f"Configuration:")
    logger.info(f"  Model: {args.model}")
    logger.info(f"  Batch Size: {args.batch_size}")
    logger.info(f"  Tensor Parallel Size: {args.tensor_parallel_size}")
    logger.info(f"  Data Parallel Size: {args.data_parallel_size}")
    logger.info(f"  Max Model Length: {args.max_model_len}")
    logger.info(f"  Output Suffix: {args.output_suffix}")
    logger.info("=" * 80)

    # Find all CSV files
    csv_files = find_csv_files(args.paths)

    if not csv_files:
        logger.error("No CSV files found!")
        return

    logger.info(f"Will process {len(csv_files)} CSV file(s)")

    # Initialize VLLM
    logger.info(f"\nInitializing VLLM with model: {args.model}")
    logger.info("This may take a few minutes on first run (downloading model)...")

    # Build VLLM initialization kwargs
    vllm_kwargs = {
        'model': args.model,
        'tensor_parallel_size': args.tensor_parallel_size,
        'max_model_len': args.max_model_len,
        'trust_remote_code': True,
        'enforce_eager': False
    }

    # Note: data_parallel_size requires multi-process setup and is not supported
    # in single-process mode. Use tensor_parallel_size instead for multiple GPUs.
    if args.data_parallel_size > 1:
        logger.warning(
            f"Data parallelism (--data-parallel-size={args.data_parallel_size}) "
            "is not supported in single-process mode. "
            "Using tensor-parallel-size instead. "
            "For data parallelism, you need a multi-process setup."
        )
        # Override with tensor parallelism
        vllm_kwargs['tensor_parallel_size'] = args.data_parallel_size
        logger.info(f"Using tensor parallelism with {args.data_parallel_size} GPUs instead")

    llm = LLM(**vllm_kwargs)
    logger.info("✓ VLLM initialized successfully")

    # Set sampling parameters
    sampling_params = SamplingParams(
        temperature=0.0,  # Deterministic
        max_tokens=20,    # We only need "A" or "B"
        top_p=1.0
    )
    logger.info(f"Sampling parameters: temperature=0.0, max_tokens=20, top_p=1.0")

    # Process each CSV file
    for file_num, csv_path in enumerate(csv_files, 1):
        try:
            logger.info(f"\n{'='*80}")
            logger.info(f"FILE {file_num}/{len(csv_files)}")
            logger.info(f"{'='*80}")

            # Process the file
            df_result = process_csv_file(csv_path, llm, sampling_params, args.batch_size)

            # Save the result
            output_path = csv_path.parent / f"{csv_path.stem}{args.output_suffix}{csv_path.suffix}"
            logger.info(f"Saving results to: {output_path}")
            df_result.to_csv(output_path, index=False)
            logger.info("✓ File saved successfully")

            # Compute and print statistics
            logger.info("\nComputing statistics...")
            stats_dict = compute_statistics(df_result)
            print_statistics(stats_dict, filename=csv_path.name)

        except Exception as e:
            logger.error(f"Error processing {csv_path}: {e}")
            import traceback
            traceback.print_exc()
            continue

    logger.info("\n" + "=" * 80)
    logger.info("✓ All files processed successfully!")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()