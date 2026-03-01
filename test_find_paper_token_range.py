#!/usr/bin/env python3
"""
Standalone test of _find_paper_token_range logic.
No GPU required — only loads the tokenizer.

Usage:
    python test_find_paper_token_range.py [--num_samples 5] [--marker $'\n\n# ']
"""
import bisect
import json
import argparse

from transformers import AutoTokenizer

DATASET_PATH = (
    "/scratch/gpfs/ZHUANGL/jl0796/shared/data"
    "/iclr_2020_2023_2025_85_5_10_balanced_original_vision_v7_filtered_test"
    "/data.json"
)
MODEL_PATH = "/scratch/gpfs/ZHUANGL/jl0796/shared/saves/bz16_lr1e-6_vision"

ENDING_MARKER = "<|im_end|>\n<|im_start|>assistant\n"


def find_paper_token_range(tokenizer, prompt_ids: list[int], paper_start_marker: str):
    """
    Exact copy of the fixed _find_paper_token_range logic from trainer.py.
    Returns: (paper_start_token, paper_length, ending_marker_found, chars, tok_char_starts)
    """
    chars = ""
    tok_char_starts: list[int] = []
    for tok_id in prompt_ids:
        tok_char_starts.append(len(chars))
        chars += tokenizer.decode([tok_id], skip_special_tokens=False)

    marker_pos = chars.find(paper_start_marker)
    if marker_pos == -1:
        return 0, len(prompt_ids), False, chars, tok_char_starts

    paper_start_char = marker_pos + len(paper_start_marker) - len(paper_start_marker.lstrip())
    paper_start_token = bisect.bisect_left(tok_char_starts, paper_start_char)
    paper_start_token = min(paper_start_token, len(prompt_ids))

    ending_pos = chars.find(ENDING_MARKER)
    ending_found = ending_pos != -1
    if ending_found:
        paper_end_token = bisect.bisect_right(tok_char_starts, ending_pos)
        paper_end_token = min(paper_end_token, len(prompt_ids)) - 1  # Exclude trailing <|im_end|>
        return paper_start_token, paper_end_token - paper_start_token, True, chars, tok_char_starts
    else:
        return paper_start_token, len(prompt_ids) - paper_start_token, False, chars, tok_char_starts


def tok_range_text(tokenizer, prompt_ids, start, end):
    """Decode a slice of token IDs, showing repr of the result."""
    return repr(tokenizer.decode(prompt_ids[start:end], skip_special_tokens=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--marker", type=str, default="\n\n# ",
                        help="Paper start marker (default: two newlines + '# ')")
    parser.add_argument("--context_tokens", type=int, default=8,
                        help="How many tokens to show around the boundary")
    args = parser.parse_args()

    print(f"Loading tokenizer from {MODEL_PATH} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    print(f"Tokenizer loaded: {type(tokenizer).__name__}\n")

    print(f"Loading dataset from {DATASET_PATH} ...")
    with open(DATASET_PATH) as f:
        dataset = json.load(f)
    print(f"Dataset size: {len(dataset)} samples\n")

    samples = dataset[: args.num_samples]

    for i, sample in enumerate(samples):
        sid = sample.get("_metadata", {}).get("submission_id", f"sample_{i:05d}")
        answer = sample.get("_metadata", {}).get("answer", "?")
        print("=" * 72)
        print(f"Sample {i}: {sid}  (answer={answer})")
        print("=" * 72)

        # Build messages: system + human only (exclude gpt turn = response)
        messages = []
        for turn in sample["conversations"]:
            if turn["from"] == "gpt":
                break
            role = "system" if turn["from"] == "system" else "user"
            # Strip <image> tokens — we are testing the text variant
            text = turn["value"].replace("<image>", "")
            messages.append({"role": role, "content": text})

        # Apply qwen2_vl chat template to get prompt text
        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # Tokenize (no special tokens added again since template already includes them)
        prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)

        print(f"Prompt length: {len(prompt_ids)} tokens")
        print(f"Marker: {repr(args.marker)}")

        paper_start, paper_length, ending_found, chars, tok_char_starts = find_paper_token_range(
            tokenizer, prompt_ids, args.marker
        )
        paper_end = paper_start + paper_length

        # --- Report marker search ---
        marker_pos = chars.find(args.marker)
        if marker_pos == -1:
            print(f"\n  [!] START MARKER NOT FOUND — returned (0, {len(prompt_ids)})")
        else:
            print(f"\n  Start marker found at char {marker_pos}")

        ending_pos_actual = chars.find(ENDING_MARKER)
        print(f"  Ending marker found: {ending_found}  pos={ending_pos_actual}")

        # --- Report return values ---
        print(f"\n  paper_start_token : {paper_start}")
        print(f"  paper_length      : {paper_length}")
        print(f"  paper_end_token   : {paper_end}  (= start + length)")

        # --- Boundary context ---
        C = args.context_tokens
        before_start = tok_range_text(tokenizer, prompt_ids,
                                      max(0, paper_start - C), paper_start)
        first_paper = tok_range_text(tokenizer, prompt_ids,
                                     paper_start, min(paper_end, paper_start + 15))
        last_paper = tok_range_text(tokenizer, prompt_ids,
                                    max(paper_start, paper_end - C), paper_end)
        after_end = tok_range_text(tokenizer, prompt_ids,
                                   paper_end, min(len(prompt_ids), paper_end + C))

        print(f"\n  {C} tokens BEFORE paper start : {before_start}")
        print(f"  First 15 paper tokens        : {first_paper}")
        print(f"  Last {C} paper tokens         : {last_paper}")
        print(f"  {C} tokens AFTER paper end    : {after_end}")
        print()


if __name__ == "__main__":
    main()
