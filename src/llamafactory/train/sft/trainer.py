# Copyright 2025 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/trainer_seq2seq.py
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import bisect
import json
import os
import gc
from types import MethodType
from typing import TYPE_CHECKING, Any, Optional, Union

import numpy as np
import torch
import cv2
from PIL import Image
from transformers import Seq2SeqTrainer
from typing_extensions import override

from ...extras import logging
from ...extras.constants import IGNORE_INDEX
from ...extras.packages import is_transformers_version_greater_than
from ..callbacks import SaveProcessorCallback
from ..fp8_utils import configure_fp8_environment, patch_accelerator_for_fp8, verify_fp8_status
from ..trainer_utils import create_custom_optimizer, create_custom_scheduler
from .train_accuracy import TrainAccuracyTracker, create_output_format_handler


if TYPE_CHECKING:
    from torch.utils.data import Dataset
    from transformers import PreTrainedTokenizer, ProcessorMixin
    from transformers.trainer import PredictionOutput

    from ...hparams import FinetuningArguments, ModelArguments


logger = logging.get_logger(__name__)


def sink_normalise(arr):
    centered = arr - arr.mean()
    clipped = np.clip(centered, 0, None)
    if clipped.max() > 1e-8:
        return clipped / clipped.max()
    return (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)


def draw_patch_grid(frame_bgr, h_patches, w_patches, color=(200, 200, 200), thickness=1):
    h, w = frame_bgr.shape[:2]
    for i in range(1, w_patches):
        x = int(round(w * i / w_patches))
        cv2.line(frame_bgr, (x, 0), (x, h - 1), color, thickness)
    for i in range(1, h_patches):
        y = int(round(h * i / h_patches))
        cv2.line(frame_bgr, (0, y), (w - 1, y), color, thickness)
    return frame_bgr


def gen_cam(image_bgr_float, mask):
    h, w = image_bgr_float.shape[:2]
    mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
    heatmap = cv2.applyColorMap(np.uint8(255 * mask_resized), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255.0
    cam = 0.5 * heatmap + 0.5 * image_bgr_float
    cam = cam / (cam.max() + 1e-8)
    return np.uint8(255 * cam)


class CustomSeq2SeqTrainer(Seq2SeqTrainer):
    r"""Inherits Seq2SeqTrainer to compute generative metrics such as BLEU and ROUGE."""

    def __init__(
        self,
        finetuning_args: "FinetuningArguments",
        processor: Optional["ProcessorMixin"],
        model_args: Optional["ModelArguments"] = None,
        gen_kwargs: Optional[dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        kwargs["processing_class"] = kwargs.pop("tokenizer")
        # Configure FP8 environment if enabled
        training_args = kwargs.get("args")
        if training_args.fp8:
            configure_fp8_environment(training_args)
            if getattr(training_args, "fp8_backend", "auto") == "te":
                patch_accelerator_for_fp8()

        super().__init__(**kwargs)
        if processor is not None:
            # avoid wrong loss under gradient accumulation
            # https://github.com/huggingface/transformers/pull/36044#issuecomment-2746657112
            self.model_accepts_loss_kwargs = False

        self.finetuning_args = finetuning_args
        if gen_kwargs is not None:
            # https://github.com/huggingface/transformers/blob/v4.45.0/src/transformers/trainer_seq2seq.py#L287
            self._gen_kwargs = gen_kwargs

        if processor is not None:
            self.add_callback(SaveProcessorCallback(processor))

        if finetuning_args.use_badam:
            from badam import BAdamCallback, clip_grad_norm_old_version  # type: ignore

            self.accelerator.clip_grad_norm_ = MethodType(clip_grad_norm_old_version, self.accelerator)
            self.add_callback(BAdamCallback)

        if finetuning_args.use_dft_loss:
            from ..trainer_utils import dft_loss_func

            self.compute_loss_func = dft_loss_func

        if training_args.fp8 and hasattr(self, "accelerator"): # verify FP8 status after trainer initialization
            verify_fp8_status(self.accelerator, training_args)

        # Counter for sequential sample IDs in attention visualization
        self._attn_viz_sample_idx = 0

        # Initialize train accuracy tracker if enabled
        self._train_accuracy_tracker = None
        if finetuning_args.sft_train_accuracy:
            handler = create_output_format_handler(
                format_type=finetuning_args.sft_train_accuracy_format,
                tokenizer=self.processing_class,
                positive_token=finetuning_args.sft_positive_token,
                negative_token=finetuning_args.sft_negative_token,
            )
            self._train_accuracy_tracker = TrainAccuracyTracker(handler)
            logger.info_rank0(
                f"SFT train accuracy tracking enabled with format '{finetuning_args.sft_train_accuracy_format}', "
                f"positive='{finetuning_args.sft_positive_token}', negative='{finetuning_args.sft_negative_token}'"
            )

    @override
    def create_optimizer(self) -> "torch.optim.Optimizer":
        if self.optimizer is None:
            self.optimizer = create_custom_optimizer(self.model, self.args, self.finetuning_args)
        return super().create_optimizer()

    @override
    def create_scheduler(
        self, num_training_steps: int, optimizer: Optional["torch.optim.Optimizer"] = None
    ) -> "torch.optim.lr_scheduler.LRScheduler":
        create_custom_scheduler(self.args, num_training_steps, optimizer)
        return super().create_scheduler(num_training_steps, optimizer)

    @override
    def _get_train_sampler(self, *args, **kwargs) -> Optional["torch.utils.data.Sampler"]:
        if self.finetuning_args.disable_shuffling:
            return torch.utils.data.SequentialSampler(self.train_dataset)

        return super()._get_train_sampler(*args, **kwargs)

    @override
    def compute_loss(
        self,
        model: "torch.nn.Module",
        inputs: dict[str, Union["torch.Tensor", Any]],
        return_outputs: bool = False,
        **kwargs,
    ) -> Union["torch.Tensor", tuple["torch.Tensor", Any]]:
        # Store input_ids and labels BEFORE parent modifies inputs
        input_ids_for_tracking = None
        labels_for_tracking = None
        if self._train_accuracy_tracker is not None:
            input_ids_for_tracking = inputs.get("input_ids")
            labels_for_tracking = inputs.get("labels")

        # Compute loss using parent class
        outputs = super().compute_loss(model, inputs, return_outputs=True, **kwargs)

        if isinstance(outputs, tuple):
            loss, model_outputs = outputs
        else:
            loss = outputs
            model_outputs = None

        # Track accuracy if enabled - use memory-efficient sliced logits approach
        if self._train_accuracy_tracker is not None and model_outputs is not None:
            with torch.no_grad():
                logits = model_outputs.logits if hasattr(model_outputs, "logits") else model_outputs.get("logits") if isinstance(model_outputs, dict) else None
                if logits is not None and input_ids_for_tracking is not None and labels_for_tracking is not None:
                    # 1. Find decision positions FIRST (no logits needed, just labels)
                    decision_positions = self._train_accuracy_tracker.handler.find_decision_positions(
                        input_ids_for_tracking, labels_for_tracking
                    )

                    # 2. IMMEDIATELY slice logits at decision positions to save memory
                    # logits: [B, S, V] → sliced_logits: [B, V]
                    batch_indices = torch.arange(logits.size(0), device=logits.device)
                    safe_positions = decision_positions.clamp(min=0)
                    sliced_logits = logits[batch_indices, safe_positions, :]  # [B, V]

                    # 3. Detach and move to CPU immediately to free GPU memory
                    sliced_logits = sliced_logits.detach().cpu()
                    decision_positions = decision_positions.detach().cpu()
                    labels_cpu = labels_for_tracking.detach().cpu()

                    # 4. Delete reference to full logits to allow GC before backward pass
                    del logits
                    if hasattr(model_outputs, "logits"):
                        model_outputs.logits = None

                    # 5. Track accuracy with sliced logits (on CPU)
                    self._train_accuracy_tracker.update_from_sliced_logits(
                        sliced_logits=sliced_logits,
                        decision_positions=decision_positions,
                        labels=labels_cpu,
                        is_training=model.training,
                    )

        if return_outputs:
            return loss, model_outputs
        return loss

    @override
    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        r"""Log metrics including aggregated SFT train accuracy metrics.

        Aggregates stored batch metrics and adds them to logs before calling parent.
        """
        # Aggregate train accuracy metrics if tracker is enabled
        if self._train_accuracy_tracker is not None:
            # Determine phase based on what's in logs
            train_eval = "train" if "loss" in logs else "eval"

            if self._train_accuracy_tracker.has_metrics(train_eval):
                accuracy_metrics = self._train_accuracy_tracker.aggregate_and_reset(
                    train_eval, self.accelerator
                )
                logs.update(accuracy_metrics)

        # Call parent to actually write logs
        super().log(logs, start_time)

    @override
    def prediction_step(
        self,
        model: "torch.nn.Module",
        inputs: dict[str, Union["torch.Tensor", Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[list[str]] = None,
        **gen_kwargs,
    ) -> tuple[Optional[float], Optional["torch.Tensor"], Optional["torch.Tensor"]]:
        r"""Prediction step with optional SFT accuracy tracking.

        When accuracy tracker is enabled and not generating:
        - Just do forward pass to get logits (no loss computation)
        - Track accuracy metrics from logits
        """
        # Attention visualization path (vision)
        if getattr(self.finetuning_args, "visualize_attention", False) and self.args.predict_with_generate:
            return self._prediction_step_with_attention_viz(
                model, inputs, prediction_loss_only, ignore_keys, **gen_kwargs
            )

        # Attention visualization path (text)
        if getattr(self.finetuning_args, "visualize_attention_text", False) and self.args.predict_with_generate:
            return self._prediction_step_with_attention_viz_text(
                model, inputs, prediction_loss_only, ignore_keys, **gen_kwargs
            )

        # Fast path for accuracy tracking: just forward pass, no loss
        if self._train_accuracy_tracker is not None and not self.args.predict_with_generate:
            input_ids = inputs.get("input_ids")
            labels = inputs.get("labels")

            with torch.no_grad():
                # 1. Find decision positions FIRST (no logits needed, just labels)
                decision_positions = self._train_accuracy_tracker.handler.find_decision_positions(input_ids, labels)

                # 2. Forward pass WITHOUT labels → no loss computation
                inputs_no_labels = {k: v for k, v in inputs.items() if k != "labels"}
                outputs = model(**inputs_no_labels)

                # 3. IMMEDIATELY slice logits at decision positions to save memory
                # logits: [B, S, V] → sliced_logits: [B, V]
                logits = outputs.logits
                batch_indices = torch.arange(logits.size(0), device=logits.device)
                # Clamp positions to valid range (handle -1 for invalid samples)
                safe_positions = decision_positions.clamp(min=0)
                sliced_logits = logits[batch_indices, safe_positions, :]  # [B, V]

                # 4. Delete full logits to free memory
                del logits, outputs
                torch.cuda.empty_cache()

                # 5. Track accuracy with sliced logits
                self._train_accuracy_tracker.update_from_sliced_logits(
                    sliced_logits=sliced_logits,
                    decision_positions=decision_positions,
                    labels=labels,
                    is_training=False,
                )

            return None, None, None

        # Standard path: generation or no accuracy tracking
        if self.args.predict_with_generate:
            labels = inputs.pop("labels", None)
        else:
            labels = inputs.get("labels")

        loss, generated_tokens, _ = super().prediction_step(
            model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys, **gen_kwargs
        )

        if generated_tokens is not None and self.args.predict_with_generate:
            generated_tokens[:, : inputs["input_ids"].size(-1)] = self.processing_class.pad_token_id
            generated_tokens = generated_tokens.contiguous()

        return loss, generated_tokens, labels

    def _prediction_step_with_attention_viz(
        self,
        model: "torch.nn.Module",
        inputs: dict[str, Union["torch.Tensor", Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[list[str]] = None,
        **gen_kwargs,
    ) -> tuple[Optional[float], Optional["torch.Tensor"], Optional["torch.Tensor"]]:
        r"""Perform attention visualization as described in simple_layerwise_attn_batch.py"""
        # Determine output directory
        output_root = self.finetuning_args.attention_viz_output_dir
        if output_root is None:
            output_root = os.path.join(self.args.output_dir, "attention_viz")
        os.makedirs(output_root, exist_ok=True)

        batch_size = inputs["input_ids"].size(0)
        device = inputs["input_ids"].device

        # Find layers (Llama, Qwen, etc.)
        layers = []
        if hasattr(model, "model") and hasattr(model.model, "language_model") and hasattr(model.model.language_model, "layers"):
            layers = model.model.language_model.layers
        elif hasattr(model, "model") and hasattr(model.model, "layers"):
            layers = model.model.layers
        elif hasattr(model, "language_model") and hasattr(model.language_model, "model") and hasattr(model.language_model.model, "layers"):
            layers = model.language_model.model.layers
        elif hasattr(model, "layers"):
            layers = model.layers

        if not layers:
            logger.warning_rank0("Could not find layers for attention visualization. Skipping.")
            return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys, **gen_kwargs)

        num_layers = len(layers)
        image_token_id = getattr(model.config, "image_token_id", None)
        if image_token_id is None and hasattr(model.config, "vision_config"):
            image_token_id = getattr(model.config.vision_config, "image_token_id", None)
        
        # Qwen2.5-VL specific
        if image_token_id is None:
            image_token_id = getattr(model.config, "vision_token_id", None)
            
        logger.info_rank0(f"Using image_token_id: {image_token_id}")

        spatial_merge = 1
        if hasattr(model.config, "vision_config"):
            spatial_merge = getattr(model.config.vision_config, "spatial_merge_size", 1)
        
        logger.info_rank0(f"Using spatial_merge: {spatial_merge}")

        for i in range(batch_size):
            # Extract single sample inputs
            sample_inputs = {}
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor):
                    if v.dim() > 0 and v.size(0) == batch_size:
                        sample_inputs[k] = v[i : i + 1]
                    else:
                        sample_inputs[k] = v
                else:
                    sample_inputs[k] = v

            # Fix 4: Truncate input_ids/attention_mask to prompt-only.
            # Only needed in causal-LM format where input_ids contains the full sequence
            # (prompt+response) and labels has IGNORE_INDEX for prompt tokens.
            # In Seq2Seq/predict_with_generate mode, input_ids is already prompt-only
            # and labels contains only the response tokens (different length) — skip truncation.
            labels_batch = inputs.get("labels")
            if labels_batch is not None and labels_batch.dim() == 2:
                labels_i = labels_batch[i]
                input_len = sample_inputs["input_ids"].shape[1]
                labels_len = labels_i.shape[0]
                if labels_len == input_len:
                    # Causal-LM format: same length, IGNORE_INDEX marks prompt
                    response_starts = (labels_i != IGNORE_INDEX).nonzero(as_tuple=True)[0]
                    if len(response_starts) > 0:
                        prompt_end = response_starts[0].item()
                        if prompt_end > 0:
                            sample_inputs["input_ids"] = sample_inputs["input_ids"][:, :prompt_end]
                            if "attention_mask" in sample_inputs and sample_inputs["attention_mask"].dim() == 2:
                                sample_inputs["attention_mask"] = sample_inputs["attention_mask"][:, :prompt_end]
                            logger.info_rank0(f"Truncated to prompt-only: {labels_len} → {prompt_end} tokens")
                else:
                    # Seq2Seq format: input_ids is already prompt-only, labels is response-only
                    logger.info_rank0(f"Seq2Seq format detected: input_ids={input_len}, labels={labels_len} — no truncation needed")

            if i == 0:
                for k, v in inputs.items():
                    if isinstance(v, torch.Tensor):
                        logger.info_rank0(f"Input {k} shape: {v.shape}")
                    else:
                        logger.info_rank0(f"Input {k} type: {type(v)}")

            # Metadata extraction
            paper_id = None
            metadata = inputs.get("_metadata", [])
            if metadata and i < len(metadata) and metadata[i] is not None:
                paper_id = metadata[i].get("paper_id") or metadata[i].get("submission_id")

            if paper_id is None:
                if "submission_id" in inputs:
                    paper_id = inputs["submission_id"][i]
                elif "paper_id" in inputs:
                    paper_id = inputs["paper_id"][i]

            if paper_id is None:
                logger.info_rank0(f"No submission_id in metadata (metadata[{i}]={metadata[i] if metadata and i < len(metadata) else 'N/A'}); using sequential ID")
                paper_id = f"sample_{self._attn_viz_sample_idx:05d}"

            image_paths = []
            if "image_paths" in inputs:
                image_paths = inputs["image_paths"][i]
            elif "images" in inputs and isinstance(inputs["images"][i], (list, tuple)) and isinstance(inputs["images"][i][0], str):
                image_paths = inputs["images"][i]

            if not image_paths and metadata and i < len(metadata) and metadata[i] is not None:
                image_paths = metadata[i].get("images") or metadata[i].get("image_paths") or []

            # Prepare image grid information
            prompt_ids = sample_inputs["input_ids"][0]
            image_positions = []
            if image_token_id is not None:
                image_positions = (prompt_ids == image_token_id).nonzero(as_tuple=True)[0].cpu().numpy()
            
            logger.info_rank0(f"Found {len(image_positions)} image tokens at positions: {image_positions}")

            grid_thw = sample_inputs.get("image_grid_thw")
            image_sizes_merged = []
            if grid_thw is not None:
                actual_grid_thw = grid_thw[0] if grid_thw.dim() == 3 else grid_thw
                logger.info_rank0(f"Image grid thw rows: {actual_grid_thw.size(0)}")
                for j in range(actual_grid_thw.size(0)):
                    h = actual_grid_thw[j, 1].item() // spatial_merge
                    w = actual_grid_thw[j, 2].item() // spatial_merge
                    image_sizes_merged.append((h, w))
            
            logger.info_rank0(f"Image sizes merged: {image_sizes_merged}")

            # Load images for OpenCV
            standardized_bgr = []
            if image_paths:
                cv2_imgs = []
                for p in image_paths:
                    if not os.path.isabs(p):
                        actual_rel = p.replace("data/images/", "")
                        p = os.path.join(self.finetuning_args.attention_viz_image_root, actual_rel)
                    if os.path.exists(p):
                        img = cv2.imread(p)
                        if img is not None:
                            cv2_imgs.append(img)
                    else:
                        logger.warning_rank0(f"Image path {p} not found for visualization.")

                if cv2_imgs:
                    max_w = max(img.shape[1] for img in cv2_imgs)
                    for img in cv2_imgs:
                        if img.shape[1] != max_w:
                            h_new = int(img.shape[0] * max_w / img.shape[1])
                            img = cv2.resize(img, (max_w, h_new), interpolation=cv2.INTER_AREA)
                        standardized_bgr.append(img)

            # Hooks setup
            attention_store = {}

            def get_hook(name):
                def hook(module, input, output):
                    if isinstance(output, tuple) and len(output) > 1 and output[1] is not None:
                        attention_store[name] = output[1].detach().cpu()
                        # Immediately free the attention tensor from GPU: replace output[1]
                        # with None so it is not also accumulated in outputs.attentions.
                        # Without this, all 28 layers' [1, H, S, S] matrices stay on GPU
                        # simultaneously (~157 GB for 10k-token sequences).
                        output = (output[0], None) + output[2:]
                    return output

                return hook

            hooks = [layers[idx].self_attn.register_forward_hook(get_hook(f"layer_{idx}")) for idx in range(num_layers)]

            # Generation parameters
            max_new_tokens = self.finetuning_args.attention_viz_max_new_tokens
            generated_ids = []
            paper_dir = os.path.join(output_root, paper_id)
            os.makedirs(paper_dir, exist_ok=True)

            # Read generation config from model (temperature, repetition_penalty, do_sample)
            gen_cfg = getattr(model, "generation_config", None)
            gen_temperature = float(getattr(gen_cfg, "temperature", 1.0)) if gen_cfg is not None else 1.0
            gen_rep_penalty = float(getattr(gen_cfg, "repetition_penalty", 1.0)) if gen_cfg is not None else 1.0
            gen_do_sample = bool(getattr(gen_cfg, "do_sample", False)) if gen_cfg is not None else False
            logger.info_rank0(
                f"Generation config: temperature={gen_temperature}, "
                f"repetition_penalty={gen_rep_penalty}, do_sample={gen_do_sample}"
            )

            def _sample_next_token(logits_last: "torch.Tensor", prev_ids: list[int]) -> int:
                """Apply repetition penalty + temperature, then greedy or sample."""
                logits = logits_last.float().clone()  # [1, vocab]
                # Repetition penalty: down-weight tokens already generated
                if gen_rep_penalty != 1.0 and prev_ids:
                    for tid in set(prev_ids):
                        if logits[0, tid] > 0:
                            logits[0, tid] /= gen_rep_penalty
                        else:
                            logits[0, tid] *= gen_rep_penalty
                # Temperature
                if gen_temperature > 0 and gen_temperature != 1.0:
                    logits = logits / gen_temperature
                # Sample or greedy
                if gen_do_sample and gen_temperature > 0:
                    probs = torch.softmax(logits, dim=-1)
                    return torch.multinomial(probs, num_samples=1).item()
                return torch.argmax(logits, dim=-1).item()

            sample_inputs.pop("labels", None)

            logger.info_rank0(f"Visualizing attention for {paper_id}...")

            # Prefill phase
            with torch.no_grad():
                with self.compute_loss_context_manager():
                    sample_inputs["output_attentions"] = True
                    sample_inputs["use_cache"] = True
                    # Cast floating point inputs to model dtype to avoid dtype mismatch
                    model_dtype = next(model.parameters()).dtype
                    for k, v in sample_inputs.items():
                        if isinstance(v, torch.Tensor) and torch.is_floating_point(v):
                            sample_inputs[k] = v.to(model_dtype)

                    outputs = model(**sample_inputs)
                    next_token_id = _sample_next_token(outputs.logits[:, -1, :], generated_ids)
                    generated_ids.append(next_token_id)
                    past_key_values = outputs.past_key_values

            # Token-by-token phase
            for step_idx in range(max_new_tokens):
                cur_token_id = generated_ids[-1]
                token_text = self.processing_class.decode([cur_token_id]).strip()
                if not token_text:
                    token_text = f"token_{cur_token_id}"

                token_safe = "".join(c for c in token_text if c.isalnum() or c in ("_", "-")).strip() or f"token_{cur_token_id}"
                save_per_layer = self.finetuning_args.attention_viz_save_per_layer
                if save_per_layer:
                    step_token_dir = os.path.join(paper_dir, "images", f"step_{step_idx:03d}_{token_safe}")
                    os.makedirs(step_token_dir, exist_ok=True)

                a_img_sum = None
                stitched_last = None

                for layer_idx in range(num_layers):
                    attn_tensor = attention_store.get(f"layer_{layer_idx}")
                    if attn_tensor is None:
                        continue

                    # Capture attention from last token to previous tokens
                    a = attn_tensor[0, :, -1, :].mean(dim=0).float().numpy()
                    if len(image_positions) > 0 and a.shape[0] >= max(image_positions):
                        a_img = a[image_positions]
                        if a_img_sum is None:
                            a_img_sum = np.zeros_like(a_img)
                        a_img_sum += a_img

                        is_last_layer = (layer_idx == num_layers - 1)
                        if standardized_bgr and (save_per_layer or is_last_layer):
                            normed_patches = sink_normalise(a_img)
                            overlaid_results = []
                            offset = 0
                            for img_bgr, (hm, wm) in zip(standardized_bgr, image_sizes_merged):
                                num_t = hm * wm
                                if offset + num_t <= len(normed_patches):
                                    mask_2d = normed_patches[offset : offset + num_t].reshape(hm, wm)
                                    res = gen_cam(np.float32(img_bgr) / 255.0, mask_2d)
                                    overlaid_results.append(draw_patch_grid(res, hm, wm))
                                    offset += num_t

                            if overlaid_results:
                                stitched_res = cv2.vconcat(overlaid_results)
                                if save_per_layer:
                                    cv2.imwrite(os.path.join(step_token_dir, f"layer_{layer_idx:02d}.jpg"), stitched_res)
                                if is_last_layer:
                                    stitched_last = stitched_res

                # Save Aggregates (Average and Last layer)
                if a_img_sum is not None and standardized_bgr:
                    avg_dir = os.path.join(paper_dir, "average", "images")
                    os.makedirs(avg_dir, exist_ok=True)
                    normed_avg = sink_normalise(a_img_sum / num_layers)
                    overlaid_avg = []
                    offset = 0
                    for img_bgr, (hm, wm) in zip(standardized_bgr, image_sizes_merged):
                        num_t = hm * wm
                        if offset + num_t <= len(normed_avg):
                            mask_2d = normed_avg[offset : offset + num_t].reshape(hm, wm)
                            res = gen_cam(np.float32(img_bgr) / 255.0, mask_2d)
                            overlaid_avg.append(draw_patch_grid(res, hm, wm))
                            offset += num_t

                    if overlaid_avg:
                        cv2.imwrite(
                            os.path.join(avg_dir, f"step_{step_idx:03d}_{token_safe}.jpg"), cv2.vconcat(overlaid_avg)
                        )

                    if stitched_last is not None:
                        last_dir = os.path.join(paper_dir, "last", "images")
                        os.makedirs(last_dir, exist_ok=True)
                        cv2.imwrite(os.path.join(last_dir, f"step_{step_idx:03d}_{token_safe}.jpg"), stitched_last)

                if cur_token_id == self.processing_class.eos_token_id or "im_end" in token_text:
                    break

                # Forward for next token
                with torch.no_grad():
                    with self.compute_loss_context_manager():
                        outputs = model(
                            input_ids=torch.tensor([[cur_token_id]], device=device),
                            past_key_values=past_key_values,
                            use_cache=True,
                            output_attentions=True,
                        )
                        next_token_id = _sample_next_token(outputs.logits[:, -1, :], generated_ids)
                        generated_ids.append(next_token_id)
                        past_key_values = outputs.past_key_values

            # Cleanup hooks and memory for this sample
            for h in hooks:
                h.remove()
            attention_store.clear()
            del past_key_values, outputs, generated_ids
            torch.cuda.empty_cache()
            gc.collect()
            self._attn_viz_sample_idx += 1

        return None, None, None

    def _find_paper_token_range(
        self,
        prompt_ids_1d: "torch.Tensor",
        paper_start_marker: str = "\n\n# ",
    ) -> tuple[int, int]:
        r"""Return [paper_start_token, paper_end_token) in prompt_ids_1d.

        Scans character positions by decoding one token at a time so the
        boundary is exact regardless of tokenizer vocabulary.
        """
        chars = ""
        tok_char_starts: list[int] = []
        for tok_id in prompt_ids_1d.tolist():
            tok_char_starts.append(len(chars))
            chars += self.processing_class.decode([tok_id], skip_special_tokens=False)

        marker_pos = chars.find(paper_start_marker)
        if marker_pos == -1:
            logger.warning_rank0(
                f"Paper start marker {repr(paper_start_marker)} not found in prompt; using full sequence as paper"
            )
            return 0, len(prompt_ids_1d)

        # Start the paper at the first non-whitespace character of the marker (i.e. the "#")
        paper_start_char = marker_pos + len(paper_start_marker) - len(paper_start_marker.lstrip())

        # End the paper just before the last <|im_end|> (close of the user turn)
        end_marker = "<|im_end|>"
        end_pos = chars.rfind(end_marker, marker_pos)
        paper_end_char = end_pos if end_pos != -1 else len(chars)

        # bisect_left gives the first token whose start char >= target
        paper_start_token = bisect.bisect_left(tok_char_starts, paper_start_char)
        paper_end_token = bisect.bisect_left(tok_char_starts, paper_end_char)
        paper_end_token = min(paper_end_token, len(prompt_ids_1d))

        return paper_start_token, paper_end_token

    def _prediction_step_with_attention_viz_text(
        self,
        model: "torch.nn.Module",
        inputs: dict[str, Union["torch.Tensor", Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[list[str]] = None,
        **gen_kwargs,
    ) -> tuple[Optional[float], Optional["torch.Tensor"], Optional["torch.Tensor"]]:
        r"""Text attention visualization: tracks per-decode-step attention to paper text tokens.

        Produces for each sample:
          <output_dir>/<paper_id>/
            average/images/step_NNN_<token>.jpg  — 1-D heatmap bar (avg across layers)
            last/images/step_NNN_<token>.jpg      — 1-D heatmap bar (last layer only)
            summary/attention_avg.npy             — [steps, paper_tokens] float32
            summary/attention_last.npy            — [steps, paper_tokens] float32
            summary/attention_avg_2d.jpg          — 2-D heatmap: rows=steps, cols=paper_tokens
            paper_tokens.txt                      — tab-sep  global_index <TAB> decoded_text
        """
        output_root = self.finetuning_args.attention_viz_text_output_dir
        if output_root is None:
            output_root = os.path.join(self.args.output_dir, "attention_viz_text")
        os.makedirs(output_root, exist_ok=True)

        batch_size = inputs["input_ids"].size(0)
        device = inputs["input_ids"].device

        # Locate transformer layers
        layers = []
        if hasattr(model, "model") and hasattr(model.model, "language_model") and hasattr(model.model.language_model, "layers"):
            layers = model.model.language_model.layers
        elif hasattr(model, "model") and hasattr(model.model, "layers"):
            layers = model.model.layers
        elif hasattr(model, "language_model") and hasattr(model.language_model, "model") and hasattr(model.language_model.model, "layers"):
            layers = model.language_model.model.layers
        elif hasattr(model, "layers"):
            layers = model.layers

        if not layers:
            logger.warning_rank0("Could not find layers for text attention visualization. Skipping.")
            return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys, **gen_kwargs)

        num_layers = len(layers)
        paper_start_marker = self.finetuning_args.attention_viz_text_paper_start_marker

        for i in range(batch_size):
            # Extract single-sample inputs
            sample_inputs = {}
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor):
                    if v.dim() > 0 and v.size(0) == batch_size:
                        sample_inputs[k] = v[i : i + 1]
                    else:
                        sample_inputs[k] = v
                else:
                    sample_inputs[k] = v

            # Truncate to prompt-only (exclude response tokens baked into input_ids by LlamaFactory)
            labels_batch = inputs.get("labels")
            if labels_batch is not None and labels_batch.dim() == 2:
                labels_i = labels_batch[i]
                response_starts = (labels_i != IGNORE_INDEX).nonzero(as_tuple=True)[0]
                if len(response_starts) > 0:
                    prompt_end = response_starts[0].item()
                    sample_inputs["input_ids"] = sample_inputs["input_ids"][:, :prompt_end]
                    if "attention_mask" in sample_inputs and sample_inputs["attention_mask"].dim() == 2:
                        sample_inputs["attention_mask"] = sample_inputs["attention_mask"][:, :prompt_end]
                    logger.info_rank0(f"[text-viz] Truncated to prompt-only: {labels_i.shape[0]} → {prompt_end} tokens")

            # Paper ID
            paper_id = None
            metadata = inputs.get("_metadata", [])
            if metadata and i < len(metadata) and metadata[i] is not None:
                paper_id = metadata[i].get("paper_id") or metadata[i].get("submission_id")
            if paper_id is None:
                logger.info_rank0(
                    f"[text-viz] No submission_id in metadata (metadata[{i}]={metadata[i] if metadata and i < len(metadata) else 'N/A'}); using sequential ID"
                )
                paper_id = f"sample_{self._attn_viz_sample_idx:05d}"

            # Find paper token range
            prompt_ids = sample_inputs["input_ids"][0]
            paper_start_token, paper_end_token = self._find_paper_token_range(prompt_ids, paper_start_marker)
            num_paper_tokens = paper_end_token - paper_start_token
            logger.info_rank0(
                f"[text-viz] {paper_id}: paper tokens [{paper_start_token}, {paper_end_token}) = {num_paper_tokens} tokens"
            )

            if num_paper_tokens <= 0:
                logger.warning_rank0(f"[text-viz] No paper tokens found for {paper_id}, skipping.")
                self._attn_viz_sample_idx += 1
                continue

            # Save paper token reference file
            paper_dir = os.path.join(output_root, paper_id)
            os.makedirs(paper_dir, exist_ok=True)
            paper_token_ids = prompt_ids[paper_start_token:paper_end_token].tolist()
            with open(os.path.join(paper_dir, "paper_tokens.txt"), "w", encoding="utf-8") as f:
                for offset, tok_id in enumerate(paper_token_ids):
                    tok_text = self.processing_class.decode([tok_id], skip_special_tokens=True)
                    f.write(f"{paper_start_token + offset}\t{repr(tok_text)}\n")

            # Hook setup
            attention_store: dict[str, "torch.Tensor"] = {}

            def get_hook(name):
                def hook(module, input, output):
                    if isinstance(output, tuple) and len(output) > 1:
                        attention_store[name] = output[1].detach().cpu()
                return hook

            hooks = [
                layers[idx].self_attn.register_forward_hook(get_hook(f"layer_{idx}"))
                for idx in range(num_layers)
            ]

            # Prefill
            max_new_tokens = self.finetuning_args.attention_viz_max_new_tokens
            sample_inputs.pop("labels", None)
            logger.info_rank0(f"[text-viz] Visualizing text attention for {paper_id}...")

            # Read generation config (temperature, repetition_penalty, do_sample)
            gen_cfg = getattr(model, "generation_config", None)
            gen_temperature = float(getattr(gen_cfg, "temperature", 1.0)) if gen_cfg is not None else 1.0
            gen_rep_penalty = float(getattr(gen_cfg, "repetition_penalty", 1.0)) if gen_cfg is not None else 1.0
            gen_do_sample = bool(getattr(gen_cfg, "do_sample", False)) if gen_cfg is not None else False

            def _sample_next_token(logits_last: "torch.Tensor", prev_ids: list[int]) -> int:
                """Apply repetition penalty + temperature, then greedy or sample."""
                logits = logits_last.float().clone()  # [1, vocab]
                if gen_rep_penalty != 1.0 and prev_ids:
                    for tid in set(prev_ids):
                        if logits[0, tid] > 0:
                            logits[0, tid] /= gen_rep_penalty
                        else:
                            logits[0, tid] *= gen_rep_penalty
                if gen_temperature > 0 and gen_temperature != 1.0:
                    logits = logits / gen_temperature
                if gen_do_sample and gen_temperature > 0:
                    probs = torch.softmax(logits, dim=-1)
                    return torch.multinomial(probs, num_samples=1).item()
                return torch.argmax(logits, dim=-1).item()

            generated_ids: list[int] = []
            with torch.no_grad():
                with self.compute_loss_context_manager():
                    sample_inputs["output_attentions"] = True
                    sample_inputs["use_cache"] = True
                    model_dtype = next(model.parameters()).dtype
                    for k, v in sample_inputs.items():
                        if isinstance(v, torch.Tensor) and torch.is_floating_point(v):
                            sample_inputs[k] = v.to(model_dtype)
                    outputs = model(**sample_inputs)
                    next_token_id = _sample_next_token(outputs.logits[:, -1, :], generated_ids)
                    generated_ids.append(next_token_id)
                    past_key_values = outputs.past_key_values

            # Accumulators for the 2-D summary
            attention_avg_all: list[np.ndarray] = []   # per step: avg across layers
            attention_last_all: list[np.ndarray] = []  # per step: last layer only

            # Precompute output dirs to avoid repeated joins
            avg_dir = os.path.join(paper_dir, "average", "images")
            last_dir = os.path.join(paper_dir, "last", "images")
            os.makedirs(avg_dir, exist_ok=True)
            os.makedirs(last_dir, exist_ok=True)
            HEATMAP_W = min(num_paper_tokens, 1024)

            # Token-by-token generation
            for step_idx in range(max_new_tokens):
                cur_token_id = generated_ids[-1]
                token_text = self.processing_class.decode([cur_token_id]).strip() or f"token_{cur_token_id}"
                token_safe = (
                    "".join(c for c in token_text if c.isalnum() or c in ("_", "-")).strip()
                    or f"token_{cur_token_id}"
                )

                a_paper_sum: Optional[np.ndarray] = None
                a_paper_last: Optional[np.ndarray] = None

                for layer_idx in range(num_layers):
                    attn_tensor = attention_store.get(f"layer_{layer_idx}")
                    if attn_tensor is None:
                        continue
                    kv_len = attn_tensor.shape[-1]
                    if kv_len < paper_end_token:
                        continue

                    # Mean over heads; last query position → current generated token
                    a = attn_tensor[0, :, -1, :].mean(dim=0).float().numpy()  # [kv_len]
                    a_paper = a[paper_start_token:paper_end_token]             # [num_paper_tokens]

                    if a_paper_sum is None:
                        a_paper_sum = np.zeros_like(a_paper)
                    a_paper_sum += a_paper

                    if layer_idx == num_layers - 1:
                        a_paper_last = a_paper.copy()

                if a_paper_sum is not None:
                    a_paper_avg = a_paper_sum / num_layers
                    attention_avg_all.append(a_paper_avg)

                    # 1-D heatmap bar: resize to fixed width, apply colormap
                    normed_avg = sink_normalise(a_paper_avg)
                    bar_avg = cv2.applyColorMap(
                        cv2.resize(np.uint8(255 * normed_avg).reshape(1, -1), (HEATMAP_W, 64), interpolation=cv2.INTER_AREA),
                        cv2.COLORMAP_JET,
                    )
                    cv2.imwrite(os.path.join(avg_dir, f"step_{step_idx:03d}_{token_safe}.jpg"), bar_avg)

                if a_paper_last is not None:
                    attention_last_all.append(a_paper_last)

                    normed_last = sink_normalise(a_paper_last)
                    bar_last = cv2.applyColorMap(
                        cv2.resize(np.uint8(255 * normed_last).reshape(1, -1), (HEATMAP_W, 64), interpolation=cv2.INTER_AREA),
                        cv2.COLORMAP_JET,
                    )
                    cv2.imwrite(os.path.join(last_dir, f"step_{step_idx:03d}_{token_safe}.jpg"), bar_last)

                if cur_token_id == self.processing_class.eos_token_id or "im_end" in token_text:
                    break

                # Forward for next token
                with torch.no_grad():
                    with self.compute_loss_context_manager():
                        outputs = model(
                            input_ids=torch.tensor([[cur_token_id]], device=device),
                            past_key_values=past_key_values,
                            use_cache=True,
                            output_attentions=True,
                        )
                        next_token_id = _sample_next_token(outputs.logits[:, -1, :], generated_ids)
                        generated_ids.append(next_token_id)
                        past_key_values = outputs.past_key_values

            # Save summary numpy arrays and 2-D heatmap
            if attention_avg_all:
                summary_dir = os.path.join(paper_dir, "summary")
                os.makedirs(summary_dir, exist_ok=True)

                attn_avg_2d = np.stack(attention_avg_all)   # [steps, paper_tokens]
                np.save(os.path.join(summary_dir, "attention_avg.npy"), attn_avg_2d)

                if attention_last_all:
                    attn_last_2d = np.stack(attention_last_all)
                    np.save(os.path.join(summary_dir, "attention_last.npy"), attn_last_2d)

                # 2-D heatmap: row-normalise each step so intra-step distribution is visible
                row_min = attn_avg_2d.min(axis=1, keepdims=True)
                row_max = attn_avg_2d.max(axis=1, keepdims=True)
                normed_2d = (attn_avg_2d - row_min) / (row_max - row_min + 1e-8)
                SUMMARY_W = 1024
                SUMMARY_H = max(attn_avg_2d.shape[0] * 8, 64)
                heatmap_2d = cv2.applyColorMap(
                    cv2.resize(np.uint8(255 * normed_2d), (SUMMARY_W, SUMMARY_H), interpolation=cv2.INTER_NEAREST),
                    cv2.COLORMAP_JET,
                )
                cv2.imwrite(os.path.join(summary_dir, "attention_avg_2d.jpg"), heatmap_2d)

            # Cleanup
            for h in hooks:
                h.remove()
            attention_store.clear()
            del past_key_values, outputs, generated_ids
            torch.cuda.empty_cache()
            gc.collect()
            self._attn_viz_sample_idx += 1

        return None, None, None

    def save_predictions(
        self, dataset: "Dataset", predict_results: "PredictionOutput", skip_special_tokens: bool = True
    ) -> None:
        r"""Save model predictions to `output_dir`.

        A custom behavior that not contained in Seq2SeqTrainer.
        """
        if not self.is_world_process_zero():
            return

        output_prediction_file = os.path.join(self.args.output_dir, "generated_predictions.jsonl")
        logger.info_rank0(f"Saving prediction results to {output_prediction_file}")

        labels = np.where(
            predict_results.label_ids != IGNORE_INDEX, predict_results.label_ids, self.processing_class.pad_token_id
        )
        preds = np.where(
            predict_results.predictions != IGNORE_INDEX,
            predict_results.predictions,
            self.processing_class.pad_token_id,
        )

        for i in range(len(preds)):
            pad_len = np.nonzero(preds[i] != self.processing_class.pad_token_id)[0]
            if len(pad_len):  # move pad token to last
                preds[i] = np.concatenate((preds[i][pad_len[0] :], preds[i][: pad_len[0]]), axis=-1)

        decoded_inputs = self.processing_class.batch_decode(dataset["input_ids"], skip_special_tokens=False)
        decoded_preds = self.processing_class.batch_decode(preds, skip_special_tokens=skip_special_tokens)
        decoded_labels = self.processing_class.batch_decode(labels, skip_special_tokens=skip_special_tokens)

        with open(output_prediction_file, "w", encoding="utf-8") as f:
            for text, pred, label in zip(decoded_inputs, decoded_preds, decoded_labels):
                f.write(json.dumps({"prompt": text, "predict": pred, "label": label}, ensure_ascii=False) + "\n")
