#!/usr/bin/env python3
"""
Optimize system prompt AND human prefix for paper acceptance prediction using TextGrad (v2).

This script uses textual gradient descent to iteratively improve both:
- System prompt (reviewer role/approach)
- Human prefix (task instruction before each paper)

Usage:
    python optimize_prompt.py --num_epochs 20 --batch_size 16 --save_dir results/
"""

import json
from datetime import datetime
from pathlib import Path

import fire
import textgrad as tg
from textgrad.engine.openai import ChatOpenAI

from utils import (
    INITIAL_HUMAN_PREFIX,
    compute_metrics,
    extract_boxed_answer,
    format_loss_feedback,
    load_dataset,
    normalize_label,
    sample_batch,
)


# Default initial system prompt (from the dataset)
INITIAL_SYSTEM_PROMPT = "You are an expert academic reviewer tasked with evaluating research papers."


class VLLMEngine(ChatOpenAI):
    """Custom engine for vLLM servers using OpenAI-compatible API."""

    def __init__(
        self,
        model_string: str,
        base_url: str,
        system_prompt: str = "",
        is_multimodal: bool = False,
    ):
        self._base_url = base_url
        self.model_string = model_string
        self.system_prompt = system_prompt
        self.is_multimodal = is_multimodal

        from openai import OpenAI
        self.client = OpenAI(
            api_key="not-needed",
            base_url=base_url,
        )

    def generate(
        self,
        content: str,
        system_prompt: str = None,
        **kwargs,
    ) -> str:
        """Generate response using vLLM server."""
        sys_prompt = system_prompt if system_prompt else self.system_prompt

        messages = []
        if sys_prompt:
            messages.append({"role": "system", "content": sys_prompt})
        messages.append({"role": "user", "content": content})

        response = self.client.chat.completions.create(
            model=self.model_string,
            messages=messages,
            **kwargs,
        )

        return response.choices[0].message.content


def setup_engines(
    inference_model: str,
    gradient_model: str,
    inference_port: int = 8000,
    gradient_port: int = 8001,
):
    """Set up textgrad engines for local vLLM servers."""
    inference_base_url = f"http://localhost:{inference_port}/v1"
    gradient_base_url = f"http://localhost:{gradient_port}/v1"

    inference_engine = VLLMEngine(
        model_string=inference_model,
        base_url=inference_base_url,
    )
    gradient_engine = VLLMEngine(
        model_string=gradient_model,
        base_url=gradient_base_url,
    )

    tg.set_backward_engine(gradient_engine)

    return inference_engine, gradient_engine


def run_inference_batch(
    engine,
    system_prompt_var: tg.Variable,
    human_prefix_var: tg.Variable,
    batch: list[dict],
) -> list[tuple[str | None, str]]:
    """
    Run inference on a batch of samples, composing system prompt + human prefix + paper content.

    Returns list of (prediction, raw_response) tuples.
    """
    results = []

    for sample in batch:
        # Compose full human message: prefix + paper content
        full_human_message = human_prefix_var.value + "\n\n" + sample["paper_content"]

        # Create input variable (not optimizable)
        question = tg.Variable(
            full_human_message,
            role_description="Paper content with task instruction prefix",
            requires_grad=False,
        )

        try:
            # Create model with current system prompt and run inference
            model = tg.BlackboxLLM(engine, system_prompt=system_prompt_var)
            response = model(question)
            raw_response = response.value

            # Extract prediction
            boxed = extract_boxed_answer(raw_response)
            prediction = normalize_label(boxed)

            results.append((prediction, raw_response))
        except Exception as e:
            print(f"Inference error: {e}")
            results.append((None, str(e)))

    return results


def create_loss_variable(
    predictions: list[str | None],
    labels: list[str],
) -> tg.Variable:
    """Create a textgrad loss variable from prediction results."""
    feedback = format_loss_feedback(predictions, labels)

    return tg.Variable(
        feedback,
        role_description="Accuracy-based loss for paper acceptance prediction. "
                        "Both the system prompt and human prefix should be improved to increase accuracy.",
        requires_grad=False,
    )


def optimize_prompt(
    # Data settings
    data_dir: str = "../data",
    train_dataset: str = "iclr_2020_2025_85_5_10_split6_original_clean_binary_noreviews_v6_train",
    val_dataset: str = "iclr_2020_2025_85_5_10_split6_original_clean_binary_noreviews_v6_validation",
    # Model settings
    inference_model: str = "Qwen/Qwen2.5-3B-Instruct",
    gradient_model: str = "Qwen/Qwen3-8B",
    inference_port: int = 8000,
    gradient_port: int = 8001,
    # Optimization settings
    num_epochs: int = 20,
    batch_size: int = 16,
    eval_batch_size: int = 50,
    initial_system_prompt: str | None = None,
    initial_human_prefix: str | None = None,
    # Constraint settings
    max_system_prompt_chars: int = 500,
    max_human_prefix_chars: int = 800,
    # Output settings
    save_dir: str = "results",
    log_interval: int = 1,
):
    """
    Main optimization loop for system prompt AND human prefix.

    Args:
        data_dir: Path to data directory containing dataset_info.json
        train_dataset: Name of training dataset (for reference)
        val_dataset: Name of validation dataset for optimization
        inference_model: Model name/path for paper review inference
        gradient_model: Model name for gradient computation
        inference_port: Port for inference vLLM server
        gradient_port: Port for gradient vLLM server
        num_epochs: Number of optimization epochs
        batch_size: Batch size for optimization steps
        eval_batch_size: Batch size for full evaluation
        initial_system_prompt: Initial system prompt (uses default if None)
        initial_human_prefix: Initial human prefix (uses default if None)
        max_system_prompt_chars: Maximum characters for system prompt
        max_human_prefix_chars: Maximum characters for human prefix
        save_dir: Directory to save results
        log_interval: How often to log progress
    """
    # Setup
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = save_path / f"optimization_log_{timestamp}.jsonl"
    best_prompts_file = save_path / f"best_prompts_{timestamp}.json"

    print("=" * 60)
    print("TextGrad Prompt Optimization v2 (System + Human Prefix)")
    print("=" * 60)

    # Load validation data
    print(f"\nLoading validation dataset: {val_dataset}")
    val_data = load_dataset(data_dir, val_dataset)
    print(f"Loaded {len(val_data)} validation samples")

    # Setup engines
    print(f"\nSetting up engines...")
    print(f"  Inference: {inference_model} @ port {inference_port}")
    print(f"  Gradient: {gradient_model} @ port {gradient_port}")

    inference_engine, _ = setup_engines(
        inference_model=inference_model,
        gradient_model=gradient_model,
        inference_port=inference_port,
        gradient_port=gradient_port,
    )

    # Initialize prompts
    if initial_system_prompt is None:
        initial_system_prompt = INITIAL_SYSTEM_PROMPT
    if initial_human_prefix is None:
        initial_human_prefix = INITIAL_HUMAN_PREFIX

    # Create optimizable variables
    system_prompt_var = tg.Variable(
        initial_system_prompt,
        role_description="System prompt that sets the AI reviewer's role and approach for paper evaluation",
        requires_grad=True,
    )

    human_prefix_var = tg.Variable(
        initial_human_prefix,
        role_description="Task instruction prefix that appears before each paper, "
                        "specifying the task (accept/reject prediction) and output format (\\boxed{Accept} or \\boxed{Reject})",
        requires_grad=True,
    )

    print(f"\nInitial system prompt ({len(initial_system_prompt)} chars):")
    print(f"  {initial_system_prompt}")
    print(f"\nInitial human prefix ({len(initial_human_prefix)} chars):")
    print(f"  {initial_human_prefix[:100]}...")

    # Create optimizer with constraints
    constraints = [
        f"Keep the system prompt under {max_system_prompt_chars} characters",
        f"Keep the human prefix under {max_human_prefix_chars} characters",
        "The human prefix MUST retain the output format instruction: \\boxed{Accept} or \\boxed{Reject}",
        "The human prefix should clearly state the task is predicting paper acceptance",
    ]

    optimizer = tg.TGD(
        parameters=[system_prompt_var, human_prefix_var],
        constraints=constraints,
    )

    print(f"\nConstraints:")
    for c in constraints:
        print(f"  - {c}")

    # Track best results - prioritize valid_rate first, then accuracy
    best_accuracy = 0.0
    best_valid_rate = 0.0
    best_system_prompt = initial_system_prompt
    best_human_prefix = initial_human_prefix

    # Optimization loop
    print(f"\nStarting optimization for {num_epochs} epochs...")
    print("-" * 60)

    for epoch in range(num_epochs):
        # Sample batch for this epoch
        batch = sample_batch(val_data, batch_size, seed=epoch)
        labels = [s["expected_label"] for s in batch]

        # Run inference
        results = run_inference_batch(
            inference_engine,
            system_prompt_var,
            human_prefix_var,
            batch,
        )
        predictions = [r[0] for r in results]

        # Compute metrics
        metrics = compute_metrics(predictions, labels)
        accuracy = metrics["accuracy"]
        num_valid = metrics["num_valid"]
        num_total = metrics["num_total"]
        valid_rate = num_valid / num_total if num_total > 0 else 0

        # Create loss and backpropagate
        loss = create_loss_variable(predictions, labels)

        try:
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        except Exception as e:
            print(f"Optimization step failed: {e}")
            continue

        # Get updated prompts
        current_system_prompt = system_prompt_var.value
        current_human_prefix = human_prefix_var.value

        # Log progress
        if epoch % log_interval == 0:
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print(f"  Valid rate: {valid_rate:.1%} ({num_valid}/{num_total})")
            print(f"  Accuracy: {accuracy:.1%}")
            print(f"  System prompt ({len(current_system_prompt)} chars): {current_system_prompt[:80]}...")
            print(f"  Human prefix ({len(current_human_prefix)} chars): {current_human_prefix[:80]}...")

        # Log to file
        log_entry = {
            "epoch": epoch + 1,
            "batch_size": len(batch),
            "valid_rate": valid_rate,
            "num_valid": num_valid,
            "num_total": num_total,
            "accuracy": accuracy,
            "accept_f1": metrics["accept_f1"],
            "reject_f1": metrics["reject_f1"],
            "system_prompt": current_system_prompt,
            "human_prefix": current_human_prefix,
            "system_prompt_chars": len(current_system_prompt),
            "human_prefix_chars": len(current_human_prefix),
        }
        with open(log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

        # Update best - prioritize valid_rate first, then accuracy
        # Use a combined score: valid_rate * 0.5 + accuracy * 0.5 (when valid_rate >= 0.8)
        # Otherwise just use valid_rate
        if valid_rate < 0.8:
            current_score = valid_rate
            best_score = best_valid_rate if best_valid_rate < 0.8 else 0.8 + best_accuracy * 0.5
        else:
            current_score = 0.8 + accuracy * 0.5
            best_score = best_valid_rate if best_valid_rate < 0.8 else 0.8 + best_accuracy * 0.5

        if current_score > best_score:
            best_accuracy = accuracy
            best_valid_rate = valid_rate
            best_system_prompt = current_system_prompt
            best_human_prefix = current_human_prefix
            if valid_rate < 0.8:
                print(f"  *** New best valid rate: {best_valid_rate:.1%} ***")
            else:
                print(f"  *** New best accuracy: {best_accuracy:.1%} (valid: {best_valid_rate:.1%}) ***")

    # Final evaluation on larger sample
    print("\n" + "=" * 60)
    print("Final Evaluation")
    print("=" * 60)

    eval_batch = sample_batch(val_data, min(eval_batch_size, len(val_data)), seed=42)
    eval_labels = [s["expected_label"] for s in eval_batch]

    # Use best prompts for final eval
    system_prompt_var.value = best_system_prompt
    human_prefix_var.value = best_human_prefix

    eval_results = run_inference_batch(
        inference_engine,
        system_prompt_var,
        human_prefix_var,
        eval_batch,
    )
    eval_predictions = [r[0] for r in eval_results]

    final_metrics = compute_metrics(eval_predictions, eval_labels)
    final_valid_rate = final_metrics["num_valid"] / final_metrics["num_total"] if final_metrics["num_total"] > 0 else 0

    print(f"\nFinal metrics (best prompts, {len(eval_batch)} samples):")
    print(f"  Valid rate: {final_valid_rate:.1%} ({final_metrics['num_valid']}/{final_metrics['num_total']})")
    print(f"  Accuracy: {final_metrics['accuracy']:.1%}")
    print(f"  Accept F1: {final_metrics['accept_f1']:.3f}")
    print(f"  Reject F1: {final_metrics['reject_f1']:.3f}")

    # Save best prompts
    best_prompts = {
        "system_prompt": best_system_prompt,
        "human_prefix": best_human_prefix,
        "best_valid_rate": best_valid_rate,
        "best_accuracy": best_accuracy,
        "final_valid_rate": final_valid_rate,
        "final_metrics": final_metrics,
        "inference_model": inference_model,
        "gradient_model": gradient_model,
        "num_epochs": num_epochs,
        "timestamp": timestamp,
    }
    with open(best_prompts_file, "w") as f:
        json.dump(best_prompts, f, indent=2)

    print(f"\nBest prompts saved to: {best_prompts_file}")
    print(f"Optimization log saved to: {log_file}")

    print("\n" + "=" * 60)
    print("Best System Prompt:")
    print("=" * 60)
    print(best_system_prompt)

    print("\n" + "=" * 60)
    print("Best Human Prefix:")
    print("=" * 60)
    print(best_human_prefix)

    return best_prompts, final_metrics


if __name__ == "__main__":
    fire.Fire(optimize_prompt)
