#!/usr/bin/env python3
"""
Optimize system prompt for paper acceptance prediction using TextGrad.

This script uses textual gradient descent to iteratively improve the system prompt
based on validation accuracy feedback.

Usage:
    python optimize_prompt.py --num_epochs 20 --batch_size 16 --save_dir results/
"""

import json
import os
from datetime import datetime
from pathlib import Path

import fire
import textgrad as tg
from textgrad.engine.openai import ChatOpenAI
from tqdm import tqdm

from utils import (
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
        # Set environment variable for OpenAI client to use custom base URL
        # Store original values to restore later if needed
        self._base_url = base_url

        # Initialize parent with model string
        # We override _client creation below
        self.model_string = model_string
        self.system_prompt = system_prompt
        self.is_multimodal = is_multimodal

        # Create OpenAI client with custom base URL
        from openai import OpenAI
        self.client = OpenAI(
            api_key="not-needed",  # vLLM doesn't require API key
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
    inference_model: str = "Qwen/Qwen2.5-3B-Instruct",
    gradient_model: str = "Qwen/Qwen3-8B",
    inference_port: int = 8000,
    gradient_port: int = 8001,
):
    """Set up textgrad engines for local vLLM servers."""
    inference_base_url = f"http://localhost:{inference_port}/v1"
    gradient_base_url = f"http://localhost:{gradient_port}/v1"

    # Create custom engines for vLLM servers
    inference_engine = VLLMEngine(
        model_string=inference_model,
        base_url=inference_base_url,
    )
    gradient_engine = VLLMEngine(
        model_string=gradient_model,
        base_url=gradient_base_url,
    )

    # Set the backward engine for gradient computation
    tg.set_backward_engine(gradient_engine)

    return inference_engine, gradient_engine


def run_inference_batch(
    model: tg.BlackboxLLM,
    batch: list[dict],
) -> list[tuple[str | None, str]]:
    """
    Run inference on a batch of samples.

    Returns list of (prediction, raw_response) tuples.
    """
    results = []

    for sample in batch:
        human_message = sample["human_message"]

        # Create input variable (not optimizable)
        question = tg.Variable(
            human_message,
            role_description="Paper content and evaluation instruction",
            requires_grad=False,
        )

        try:
            # Run inference
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
                        "The system prompt should be improved to increase accuracy.",
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
    initial_prompt: str | None = None,
    # Output settings
    save_dir: str = "results",
    log_interval: int = 1,
):
    """
    Main optimization loop for system prompt.

    Args:
        data_dir: Path to data directory containing dataset_info.json
        train_dataset: Name of training dataset (for reference)
        val_dataset: Name of validation dataset for optimization
        inference_model: Model name for paper review inference
        gradient_model: Model name for gradient computation
        inference_port: Port for inference vLLM server
        gradient_port: Port for gradient vLLM server
        num_epochs: Number of optimization epochs
        batch_size: Batch size for optimization steps
        eval_batch_size: Batch size for full evaluation
        initial_prompt: Initial system prompt (uses default if None)
        save_dir: Directory to save results
        log_interval: How often to log progress
    """
    # Setup
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = save_path / f"optimization_log_{timestamp}.jsonl"
    best_prompt_file = save_path / f"best_prompt_{timestamp}.txt"

    print("=" * 60)
    print("TextGrad Prompt Optimization for Paper Review")
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

    # Initialize system prompt
    if initial_prompt is None:
        initial_prompt = INITIAL_SYSTEM_PROMPT

    system_prompt = tg.Variable(
        initial_prompt,
        role_description="System prompt for paper acceptance prediction. "
                        "This prompt sets the context and approach for the AI reviewer.",
        requires_grad=True,
    )

    print(f"\nInitial system prompt:\n{initial_prompt}")

    # Create model and optimizer
    model = tg.BlackboxLLM(inference_engine, system_prompt=system_prompt)
    optimizer = tg.TGD(parameters=[system_prompt])

    # Track best results
    best_accuracy = 0.0
    best_prompt = initial_prompt

    # Optimization loop
    print(f"\nStarting optimization for {num_epochs} epochs...")
    print("-" * 60)

    for epoch in range(num_epochs):
        # Sample batch for this epoch
        batch = sample_batch(val_data, batch_size, seed=epoch)
        labels = [s["expected_label"] for s in batch]

        # Run inference
        results = run_inference_batch(model, batch)
        predictions = [r[0] for r in results]

        # Compute metrics
        metrics = compute_metrics(predictions, labels)
        accuracy = metrics["accuracy"]

        # Create loss and backpropagate
        loss = create_loss_variable(predictions, labels)

        try:
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        except Exception as e:
            print(f"Optimization step failed: {e}")
            continue

        # Get updated prompt
        current_prompt = system_prompt.value

        # Log progress
        if epoch % log_interval == 0:
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print(f"  Batch accuracy: {accuracy:.1%}")
            print(f"  Current prompt: {current_prompt[:100]}...")

        # Log to file
        log_entry = {
            "epoch": epoch + 1,
            "batch_size": len(batch),
            "accuracy": accuracy,
            "accept_f1": metrics["accept_f1"],
            "reject_f1": metrics["reject_f1"],
            "num_valid": metrics["num_valid"],
            "prompt": current_prompt,
        }
        with open(log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

        # Update best
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_prompt = current_prompt
            print(f"  New best accuracy: {best_accuracy:.1%}")

    # Final evaluation on larger sample
    print("\n" + "=" * 60)
    print("Final Evaluation")
    print("=" * 60)

    eval_batch = sample_batch(val_data, min(eval_batch_size, len(val_data)), seed=42)
    eval_labels = [s["expected_label"] for s in eval_batch]

    # Use best prompt for final eval
    system_prompt.value = best_prompt
    eval_results = run_inference_batch(model, eval_batch)
    eval_predictions = [r[0] for r in eval_results]

    final_metrics = compute_metrics(eval_predictions, eval_labels)
    print(f"\nFinal metrics (best prompt, {len(eval_batch)} samples):")
    print(f"  Accuracy: {final_metrics['accuracy']:.1%}")
    print(f"  Accept F1: {final_metrics['accept_f1']:.3f}")
    print(f"  Reject F1: {final_metrics['reject_f1']:.3f}")

    # Save best prompt
    with open(best_prompt_file, "w") as f:
        f.write(best_prompt)

    print(f"\nBest prompt saved to: {best_prompt_file}")
    print(f"Optimization log saved to: {log_file}")

    print("\n" + "=" * 60)
    print("Best System Prompt:")
    print("=" * 60)
    print(best_prompt)

    return best_prompt, final_metrics


if __name__ == "__main__":
    fire.Fire(optimize_prompt)
