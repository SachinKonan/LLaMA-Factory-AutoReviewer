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
import functools
from types import MethodType
from typing import TYPE_CHECKING, Any, Optional, Union

import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
import pandas as pd
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


def sum_normalise(arr):
    """Normalise arr to a probability distribution (sum = 1) over the tracked tokens,
    then scale by the max so the colormap uses the full [0, 1] range for display."""
    s = arr.sum()
    prob = arr / s if s > 1e-9 else arr
    m = prob.max()
    return prob / m if m > 1e-9 else prob


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
        data_args: Optional["DataArguments"] = None,
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
        self.data_args = data_args
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

        self._eval_paper_ids: Optional[list[str]] = None
        
        self._block_metadata: Optional[dict[str, list[dict[str, Any]]]] = None

    def _get_block_metadata(self, submission_id: str) -> Optional[list[dict[str, Any]]]:
        """Lazy loader for massive_metadata_v7.csv content_list_json."""
        if self._block_metadata is None:
            metadata_path = "/scratch/gpfs/ZHUANGL/jl0796/shared/data/massive_metadata_v7.csv"
            if not os.path.exists(metadata_path):
                logger.warning_rank0(f"Metadata file {metadata_path} not found.")
                self._block_metadata = {}
            else:
                try:
                    logger.info_rank0(f"Loading block metadata from {metadata_path}...")
                    df = pd.read_csv(metadata_path, usecols=["submission_id", "content_list_json"])
                    self._block_metadata = {}
                    for _, row in df.iterrows():
                        sub_id = str(row["submission_id"])
                        content = row["content_list_json"]
                        if isinstance(content, str):
                            try:
                                self._block_metadata[sub_id] = json.loads(content)
                            except json.JSONDecodeError:
                                pass
                    logger.info_rank0(f"Loaded metadata for {len(self._block_metadata)} papers.")
                except Exception as e:
                    logger.warning_rank0(f"Failed to load metadata: {e}")
                    self._block_metadata = {}
        
        return self._block_metadata.get(submission_id)

    def _get_sections_from_blocks(self, blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Group content blocks into sections by headers."""
        sections = []
        current_section = {"header": "Preamble", "blocks": []}
        for b in blocks:
            if b.get("type") == "text" and "text_level" in b:
                if current_section["blocks"] or current_section["header"] != "Preamble":
                    sections.append(current_section)
                current_section = {"header": b.get("text", "Header"), "blocks": []}
            current_section["blocks"].append(b)
        if current_section["blocks"]:
            sections.append(current_section)
        return sections

    def _get_token_to_section_mapping(self, paper_token_ids, sections) -> list[str]:
        """Map each paper token to a section header string."""
        tokens_decoded = [self.processing_class.decode([tid]) for tid in paper_token_ids]
        full_text = "".join(tokens_decoded)
        
        section_ranges = []
        last_found_idx = 0
        for section in sections:
            header = section["header"]
            idx = full_text.find(header, last_found_idx)
            if idx != -1:
                section_ranges.append({"header": header, "char_start": idx})
                last_found_idx = idx + len(header)
        
        if not section_ranges:
            return ["None"] * len(paper_token_ids)
            
        for i in range(len(section_ranges) - 1):
            section_ranges[i]["char_end"] = section_ranges[i+1]["char_start"]
        section_ranges[-1]["char_end"] = len(full_text)
        
        cum_len = 0
        token_boundaries = []
        for t_text in tokens_decoded:
            token_boundaries.append((cum_len, cum_len + len(t_text)))
            cum_len += len(t_text)
            
        mapping = ["None"] * len(paper_token_ids)
        for s in section_ranges:
            s_start = s["char_start"]
            s_end = s["char_end"]
            header = s["header"]
            for t_idx, (b_start, b_end) in enumerate(token_boundaries):
                if max(s_start, b_start) < min(s_end, b_end):
                    mapping[t_idx] = header
                    
        return mapping

    def _get_eval_paper_ids(self) -> list[str]:
        """Load paper IDs from the evaluation dataset json file."""
        if self._eval_paper_ids is not None:
            return self._eval_paper_ids
        
        self._eval_paper_ids = []
        try:
            # Try to find the dataset file via data_args or dataset_info.json
            dataset_dir = getattr(self.data_args, "dataset_dir", None) or getattr(self.finetuning_args, "dataset_dir", "/scratch/gpfs/ZHUANGL/jl0796/shared/data")
            info_path = os.path.join(dataset_dir, "dataset_info.json")
            
            # Try multiple places for eval_name
            eval_name = getattr(self.args, "eval_dataset", None)
            if not eval_name and self.data_args is not None:
                # Try both dataset and eval_dataset fields
                eval_name = getattr(self.data_args, "dataset", None) or getattr(self.data_args, "eval_dataset", None)
                if eval_name:
                    logger.info_rank0(f"[attn_viz] Found eval_name in data_args: {eval_name}")
            
            if not eval_name: # Parsing from config file if present in sys.argv
                import sys
                for arg in sys.argv:
                    if arg.endswith(".yaml") and os.path.exists(arg):
                        try:
                            import yaml
                            with open(arg, "r") as f:
                                config = yaml.safe_load(f)
                                eval_name = config.get("dataset")
                                if eval_name: break
                        except:
                            pass
            
            if isinstance(eval_name, (list, tuple)):
                eval_name = eval_name[0]
            
            logger.info_rank0(f"[attn_viz] Resolved eval_name: {eval_name}")
            
            if eval_name and os.path.exists(info_path):
                with open(info_path, "r") as f:
                    info = json.load(f)
                if eval_name in info:
                    data_file = os.path.join(dataset_dir, info[eval_name]["file_name"])
                    if os.path.exists(data_file):
                        with open(data_file, "r") as f:
                            data = json.load(f)
                            for item in data:
                                m = item.get("_metadata", {})
                                s_id = m.get("submission_id") or m.get("paper_id")
                                if s_id:
                                    self._eval_paper_ids.append(s_id)
        except Exception as e:
            logger.warning_rank0(f"Failed to load eval paper IDs: {e}")
            
        return self._eval_paper_ids

    def _compute_section_weights_vision(self, a_img, image_sizes_merged, sections):
        """Aggregate image patch attention into semantic sections using bboxes."""
        section_weights = {}
        offset = 0
        # Normalize a_img before aggregation to stay consistent with heatmaps
        sum_a = a_img.sum()
        a_norm = a_img / sum_a if sum_a > 1e-9 else a_img
        
        for page_idx, (hm, wm) in enumerate(image_sizes_merged):
            num_t = hm * wm
            if offset + num_t > len(a_norm): break
            page_a = a_norm[offset : offset + num_t].reshape(hm, wm)
            
            for section in sections:
                header = section["header"]
                if header not in section_weights:
                    section_weights[header] = {"sum": 0.0, "count": 0}
                
                for block in section["blocks"]:
                    if block.get("page_idx") == page_idx and "bbox" in block:
                        x0, y0, x1, y1 = block["bbox"]
                        # Map 0-1000 PDF coords to patch grid
                        r0 = max(0, int(y0 * hm / 1000))
                        r1 = min(hm, int(y1 * hm / 1000) + 1)
                        c0 = max(0, int(x0 * wm / 1000))
                        c1 = min(wm, int(x1 * wm / 1000) + 1)
                        
                        if r1 > r0 and c1 > c0:
                            block_slice = page_a[r0:r1, c0:c1]
                            section_weights[header]["sum"] += block_slice.sum()
                            section_weights[header]["count"] += block_slice.size
            offset += num_t
            
        return {h: float(d["sum"]/d["count"]) for h, d in section_weights.items() if d["count"] > 0}

    def _compute_section_weights_text(self, a_text, paper_token_ids, sections):
        """Aggregate text attention into semantic sections using string matching."""
        tokens_decoded = [self.processing_class.decode([tid]) for tid in paper_token_ids]
        full_text = "".join(tokens_decoded)
        
        # Normalize a_text
        sum_a = a_text.sum()
        a_norm = a_text / sum_a if sum_a > 1e-9 else a_text
        
        section_ranges = []
        last_found_idx = 0
        for section in sections:
            header = section["header"]
            # Find the header in full_text starting from last_found_idx
            idx = full_text.find(header, last_found_idx)
            if idx != -1:
                section_ranges.append({"header": header, "char_start": idx})
                last_found_idx = idx + len(header)
        
        if not section_ranges:
            return {}
            
        # Add end markers
        for i in range(len(section_ranges) - 1):
            section_ranges[i]["char_end"] = section_ranges[i+1]["char_start"]
        section_ranges[-1]["char_end"] = len(full_text)
        
        # Map char offsets back to token indices
        results = {}
        cum_len = 0
        token_boundaries = []
        for t_idx, t_text in enumerate(tokens_decoded):
            token_boundaries.append((cum_len, cum_len + len(t_text)))
            cum_len += len(t_text)
            
        for s in section_ranges:
            s_start = s["char_start"]
            s_end = s["char_end"]
            
            # Find tokens overlapping with [s_start, s_end]
            s_tokens = []
            for t_idx, (b_start, b_end) in enumerate(token_boundaries):
                # Check overlap
                if max(s_start, b_start) < min(s_end, b_end):
                    s_tokens.append(t_idx)
            
            if s_tokens:
                sub_a = a_norm[s_tokens]
                results[s.get("header")] = float(sub_a.mean())
                
        return results

    # def _save_semantic_heatmaps(self, paper_id, section_weights, sections, image_paths, output_dir):
    #     """Color PDF bounding boxes based on section-wise attention weights."""
    #     if not paper_id or not section_weights:
    #         return

    #     # If image_paths is missing (text-only), reconstruct them
    #     if not image_paths:
    #         image_paths = [
    #             f"data/images/{paper_id}/page_{p+1}_noreferences_original.png"
    #             for p in range(10) # Assume 10 pages or check metadata
    #         ]

    #     # Prepare normalized weights for colormap
    #     all_vals = np.array(list(section_weights.values()))
    #     # Use log scale for better vividness if values are skewed
    #     # weights are already normalized to sum to 1 in compute_section_weights
    #     weights_mapped = np.log10(all_vals + 1e-9)
    #     max_v = weights_mapped.max()
    #     min_v = weights_mapped.min()
        
    #     # Color mapper
    #     import matplotlib.cm as cm
    #     colormap = cm.get_cmap("viridis")

    #     processed_pages = []
    #     for page_idx, p in enumerate(image_paths):
    #         if not os.path.isabs(p):
    #             actual_rel = p.replace("data/images/", "")
    #             p = os.path.join(self.finetuning_args.attention_viz_image_root, actual_rel)
            
    #         if not os.path.exists(p):
    #             if "_noreferences_original" not in p:
    #                 p_alt = p.replace(".png", "_noreferences_original.png")
    #                 if os.path.exists(p_alt): p = p_alt
            
    #         if not os.path.exists(p):
    #             continue
            
    #         img = cv2.imread(p)
    #         if img is None: continue
            
    #         overlay = img.copy()
    #         H, W = img.shape[:2]
    #         any_drawn = False

    #         for section in sections:
    #             header = section["header"]
    #             weight = section_weights.get(header, 0.0)
    #             if weight <= 0: continue
                
    #             val_log = np.log10(weight + 1e-9)
    #             if max_v > min_v:
    #                 norm_w = (val_log - min_v) / (max_v - min_v)
    #             else:
    #                 norm_w = 0.5
                
    #             rgba = colormap(norm_w)
    #             bgr = (int(rgba[2]*255), int(rgba[1]*255), int(rgba[0]*255))

    #             for block in section["blocks"]:
    #                 if block.get("page_idx") == page_idx and "bbox" in block:
    #                     x0, y0, x1, y1 = block["bbox"]
    #                     ix0 = int(x0 * W / 1000)
    #                     iy0 = int(y0 * H / 1000)
    #                     ix1 = int(x1 * W / 1000)
    #                     iy1 = int(y1 * H / 1000)
    #                     cv2.rectangle(overlay, (ix0, iy0), (ix1, iy1), bgr, -1)
    #                     any_drawn = True
            
    #         if any_drawn:
    #             alpha = 0.4
    #             cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
            
    #         # Standardize size for grid
    #         grid_img = cv2.resize(img, (512, 640), interpolation=cv2.INTER_AREA)
    #         processed_pages.append(grid_img)
            
    #     if processed_pages:
    #         n_pages = len(processed_pages)
    #         half_n = (n_pages + 1) // 2
    #         top_row = processed_pages[:half_n]
    #         bottom_row = processed_pages[half_n:]
            
    #         if len(bottom_row) < half_n:
    #             h, w, c = top_row[-1].shape
    #             bottom_row.append(np.zeros((h, w, c), dtype=np.uint8))
                
    #         top_stitched = cv2.hconcat(top_row)
    #         bottom_stitched = cv2.hconcat(bottom_row)
    #         stitched = cv2.vconcat([top_stitched, bottom_stitched])
            
    #         cv2.imwrite(os.path.join(output_dir, "semantic_heatmap_grid.jpg"), stitched)

    # Modified for better colors & normalization (linear)
    def _save_semantic_heatmaps(self, paper_id, section_weights, sections, image_paths, output_dir):
    """Color PDF bounding boxes based on section-wise attention weights."""

    if not paper_id or not section_weights:
        return

    # If image_paths is missing (text-only), reconstruct them
    if not image_paths:
        image_paths = [
            f"data/images/{paper_id}/page_{p+1}_noreferences_original.png"
            for p in range(10)  # Assume 10 pages or check metadata
        ]

    import numpy as np
    import os
    import cv2
    import matplotlib.cm as cm

    # Normalize weights by dividing by max
    all_vals = np.array(list(section_weights.values()))
    max_v = all_vals.max() if len(all_vals) > 0 else 1.0

    # Use viridis colormap (smooth perceptual gradient)
    colormap = cm.get_cmap("viridis")

    processed_pages = []

    for page_idx, p in enumerate(image_paths):

        if not os.path.isabs(p):
            actual_rel = p.replace("data/images/", "")
            p = os.path.join(self.finetuning_args.attention_viz_image_root, actual_rel)

        if not os.path.exists(p):
            if "_noreferences_original" not in p:
                p_alt = p.replace(".png", "_noreferences_original.png")
                if os.path.exists(p_alt):
                    p = p_alt

        if not os.path.exists(p):
            continue

        img = cv2.imread(p)
        if img is None:
            continue

        overlay = img.copy()
        H, W = img.shape[:2]
        any_drawn = False

        for section in sections:
            header = section["header"]
            weight = section_weights.get(header, 0.0)

            if weight <= 0 or max_v == 0:
                continue

            # Normalize by max
            norm_w = weight / max_v
            norm_w = np.clip(norm_w, 0.0, 1.0)

            rgba = colormap(norm_w)
            bgr = (
                int(rgba[2] * 255),
                int(rgba[1] * 255),
                int(rgba[0] * 255),
            )

            for block in section["blocks"]:
                if block.get("page_idx") == page_idx and "bbox" in block:
                    x0, y0, x1, y1 = block["bbox"]

                    ix0 = int(x0 * W / 1000)
                    iy0 = int(y0 * H / 1000)
                    ix1 = int(x1 * W / 1000)
                    iy1 = int(y1 * H / 1000)

                    cv2.rectangle(overlay, (ix0, iy0), (ix1, iy1), bgr, -1)
                    any_drawn = True

        if any_drawn:
            alpha = 0.4
            cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

        # Standardize size for grid
        grid_img = cv2.resize(img, (512, 640), interpolation=cv2.INTER_AREA)
        processed_pages.append(grid_img)

    if processed_pages:
        n_pages = len(processed_pages)
        half_n = (n_pages + 1) // 2

        top_row = processed_pages[:half_n]
        bottom_row = processed_pages[half_n:]

        if len(bottom_row) < half_n:
            h, w, c = top_row[-1].shape
            bottom_row.append(np.zeros((h, w, c), dtype=np.uint8))

        top_stitched = cv2.hconcat(top_row)
        bottom_stitched = cv2.hconcat(bottom_row)
        stitched = cv2.vconcat([top_stitched, bottom_stitched])

        cv2.imwrite(
            os.path.join(output_dir, "semantic_heatmap_grid.jpg"),
            stitched
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
        all_generated_ids: list[list[int]] = []  # collect per-sample for return value

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

            # Debug inputs and metadata
            logger.info_rank0(f"prediction_step keys: {list(inputs.keys())}")
            if "_metadata" in inputs:
                logger.info_rank0(f"Found _metadata: {inputs['_metadata']}")

            # Metadata extraction
            paper_id = None
            metadata = inputs.get("_metadata", [])
            if metadata and i < len(metadata) and metadata[i] is not None:
                if isinstance(metadata[i], dict):
                    paper_id = metadata[i].get("paper_id") or metadata[i].get("submission_id")
                elif isinstance(metadata[i], str):
                    try:
                        m_json = json.loads(metadata[i])
                        paper_id = m_json.get("paper_id") or m_json.get("submission_id")
                    except:
                        pass

            if paper_id is None:
                for k in ["submission_id", "paper_id", "id"]:
                    if k in inputs:
                        val = inputs[k][i]
                        paper_id = val if isinstance(val, str) else str(val)
                        break

            # Robust fallback: lookup directly from eval_dataset or data.json
            if paper_id is None:
                eval_ids = self._get_eval_paper_ids()
                if 0 <= self._attn_viz_sample_idx < len(eval_ids):
                    paper_id = eval_ids[self._attn_viz_sample_idx]

            if paper_id is None and hasattr(self, "eval_dataset"):
                try:
                    sample = self.eval_dataset[self._attn_viz_sample_idx]
                    m = sample.get("_metadata")
                    if m:
                        paper_id = m.get("submission_id") or m.get("paper_id")
                except:
                    pass

            image_paths = []
            if "image_paths" in inputs:
                image_paths = inputs["image_paths"][i]
            elif "images" in inputs and isinstance(inputs["images"][i], (list, tuple)) and isinstance(inputs["images"][i][0], str):
                image_paths = inputs["images"][i]

            if not image_paths and hasattr(self, "eval_dataset"):
                try:
                    sample = self.eval_dataset[self._attn_viz_sample_idx]
                    image_paths = sample.get("images") or sample.get("image_paths") or []
                except:
                    pass

            if not image_paths and metadata and i < len(metadata) and metadata[i] is not None:
                if isinstance(metadata[i], dict):
                    image_paths = metadata[i].get("images") or metadata[i].get("image_paths") or []

            # If still no paper_id, try to extract from image_paths
            if paper_id is None and image_paths:
                for p in image_paths:
                    parts = p.split("/")
                    if "images" in parts:
                        idx = parts.index("images")
                        if idx + 1 < len(parts):
                            paper_id = parts[idx+1]
                            break

            if paper_id is None:
                logger.info_rank0(f"No submission_id in metadata; using sequential ID")
                paper_id = f"sample_{self._attn_viz_sample_idx:05d}"

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

            # ── Attention capture via Q/K hooks on the last layer only ──────────
            # Strategy: run the full forward with flash attention (no output_attentions).
            # For the last transformer layer, capture Q (from q_proj, before RoPE) and
            # position_embeddings (cos, sin) via hooks.  After each forward, apply RoPE
            # to Q ourselves, read K for image positions from past_key_values, and compute
            # softmax(Q_last @ K_image.T * scaling) averaged over heads.  This never
            # materialises the full [seq, seq] attention matrix.
            last_layer = layers[num_layers - 1]
            last_attn = last_layer.self_attn

            q_cap: dict = {}   # filled by hooks each forward call

            def _q_proj_hook(module, input, output):
                # output: [1, seq_len, num_heads * head_dim]
                q_cap["q_raw"] = output.detach()

            def _self_attn_pre_hook(module, args, kwargs):
                # position_embeddings=(cos, sin) is passed as a kwarg by the model
                pe = kwargs.get("position_embeddings")
                if pe is not None:
                    q_cap["cos"] = pe[0].detach()
                    q_cap["sin"] = pe[1].detach()

            h_qproj = last_attn.q_proj.register_forward_hook(_q_proj_hook)
            h_pre   = last_attn.register_forward_pre_hook(_self_attn_pre_hook, with_kwargs=True)

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

            def _compute_a_img(past_key_values) -> Optional[torch.Tensor]:
                """After a forward pass, compute last-layer attention from the last query
                token to all image positions using captured Q and K from the KV cache.
                Uses full softmax over all tokens and returns [num_heads, n_img] tensor.
                """
                if not q_cap or "q_raw" not in q_cap or "cos" not in q_cap:
                    return None
                if len(image_positions) == 0:
                    return None
                try:
                    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import apply_multimodal_rotary_pos_emb
                    q_raw = q_cap["q_raw"]  # [1, q_len, num_heads * head_dim]
                    cos   = q_cap["cos"]
                    sin   = q_cap["sin"]
                    bsz, q_len, _ = q_raw.shape
                    num_heads  = last_attn.num_heads
                    head_dim   = last_attn.head_dim
                    num_kv_heads = last_attn.num_key_value_heads
                    scaling    = last_attn.scaling

                    # Reshape Q: [1, num_heads, q_len, head_dim]
                    q = q_raw.view(bsz, q_len, num_heads, head_dim).transpose(1, 2)

                    # Apply multimodal RoPE (same call as inside self_attn.forward)
                    mrope_section = last_attn.rope_scaling["mrope_section"]
                    q_rot, _ = apply_multimodal_rotary_pos_emb(q, q, cos, sin, mrope_section)
                    q_last = q_rot[:, :, -1:, :]                         # [1, num_heads, 1, head_dim]

                    # Full K from KV cache of the last layer
                    k_full = past_key_values[num_layers - 1][0]          # [1, kv_heads, kv_len, head_dim]
                    groups = num_heads // num_kv_heads
                    k_exp = k_full.repeat_interleave(groups, dim=1)      # [1, num_heads, kv_len, head_dim]

                    # Full attention for last query token: [1, num_heads, 1, kv_len]
                    attn_scores = torch.matmul(q_last, k_exp.transpose(-1, -2)) * scaling
                    # Full softmax to get true attention weights
                    attn_weights_full = torch.softmax(attn_scores.float(), dim=-1).to(q_last.dtype) # [1, num_heads, 1, kv_len]

                    # Slice out image positions
                    img_pos_t = torch.from_numpy(image_positions).long()
                    a_img_heads = attn_weights_full[0, :, 0, img_pos_t]   # [num_heads, n_img]
                    return a_img_heads
                except Exception as exc:
                    import traceback
                    logger.warning_rank0(f"_compute_a_img failed: {traceback.format_exc()}")
                    return None

            def _save_step(step_idx: int, token_safe: str, a_img_np: np.ndarray, is_smoothed: bool = False) -> None:
                """Render and save heatmap overlays for a single generation step."""
                if a_img is None or not standardized_bgr:
                    return
                # Normalize the ENTIRE sequence globally first so all pages share the same colormap scale
                normed_global = sum_normalise(a_img)
                suffix = "_smoothed" if is_smoothed else ""
                
                def _save_variant(normed_arr, variant_name):
                    overlaid = []
                    offset = 0
                    for img_bgr, (hm, wm) in zip(standardized_bgr, image_sizes_merged):
                        num_t = hm * wm
                        if offset + num_t <= len(normed_arr):
                            # Extract the globally-normalized patch scores for this specific image page
                            mask_2d = normed_arr[offset : offset + num_t].reshape(hm, wm)
                            res = gen_cam(np.float32(img_bgr) / 255.0, mask_2d)
                            overlaid.append(draw_patch_grid(res, hm, wm))
                            offset += num_t
                    if overlaid:
                        n_pages = len(overlaid)
                        half_n = (n_pages + 1) // 2
                        
                        top_row = overlaid[:half_n]
                        bottom_row = overlaid[half_n:]
                        
                        # Pad the bottom row if the number of pages is odd to match widths
                        if len(bottom_row) < half_n:
                            h, w, c = top_row[-1].shape
                            blank_img = np.zeros((h, w, c), dtype=np.uint8)
                            bottom_row.append(blank_img)
                            
                        top_stitched = cv2.hconcat(top_row)
                        bottom_stitched = cv2.hconcat(bottom_row)
                        stitched = cv2.vconcat([top_stitched, bottom_stitched])
                        
                        # Save to both "last/" (last-layer) and "average/" (same here — single layer)
                        for subdir in ("last", "average"):
                            out_dir = os.path.join(paper_dir, subdir, f"images{variant_name}")
                            os.makedirs(out_dir, exist_ok=True)
                            cv2.imwrite(os.path.join(out_dir, f"step_{step_idx:03d}_{token_safe}.jpg"), stitched)

                _save_variant(normed_global, f"{suffix}")
                for k in [5, 10, 25]:
                    threshold = np.percentile(normed_global, 100 - k)
                    filtered = np.where(normed_global >= threshold, normed_global, 0.0)
                    _save_variant(sum_normalise(filtered), f"_top{k}{suffix}")

            sample_inputs.pop("labels", None)
            logger.info_rank0(f"Visualizing attention for {paper_id}...")

            # ── Prefill ───────────────────────────────────────────────────────────
            # Setup for generation loop
            past_key_values = None
            cache_position = torch.arange(sample_inputs["input_ids"].shape[1], device=device)

            # ── Prefill ───────────────────────────────────────────────────────────
            with torch.no_grad():
                with self.compute_loss_context_manager():
                    model_dtype = next(model.parameters()).dtype
                    for k, v in sample_inputs.items():
                        if isinstance(v, torch.Tensor) and torch.is_floating_point(v):
                            sample_inputs[k] = v.to(model_dtype)

                    inputs_for_gen = model.prepare_inputs_for_generation(
                        **sample_inputs,
                        past_key_values=past_key_values,
                        use_cache=True,
                        cache_position=cache_position,
                    )
                    
                    outputs = model(**inputs_for_gen)
                    next_token_id = _sample_next_token(outputs.logits[:, -1, :], generated_ids)
                    generated_ids.append(next_token_id)
                    past_key_values = outputs.past_key_values

            # Maintain attention mask for generation loop manually
            attention_mask = sample_inputs.get("attention_mask")
            if attention_mask is None:
                attention_mask = torch.ones_like(sample_inputs["input_ids"])

            attention_img_all: list[np.ndarray] = []
            decision_img_all: list[tuple[int, str, np.ndarray]] = []

            # ── Token-by-token decode ─────────────────────────────────────────────
            for step_idx in range(max_new_tokens):
                cur_token_id = generated_ids[-1]
                token_text = self.processing_class.decode([cur_token_id]).strip() or f"token_{cur_token_id}"
                token_safe = "".join(c for c in token_text if c.isalnum() or c in ("_", "-")).strip() or f"token_{cur_token_id}"

                # Compute and save attention heatmap for this step's token
                a_img_heads = _compute_a_img(past_key_values)
                a_img = None
                if a_img_heads is not None:
                    print(f"a_img_heads shape: {a_img_heads.shape}")
                    a_img = a_img_heads.mean(dim=0).float().cpu().numpy()
                    print(f"a_img (averaged across heads) shape: {a_img.shape}")
                    a_img_sum = a_img_heads.sum().item() / last_attn.num_heads
                    print(f"a_img_sum: {a_img_sum}")
                    # Smoothing: average over each horizontal row of tokens
                    a_img_smoothed = np.zeros_like(a_img)
                    offset = 0
                    for hm, wm in image_sizes_merged:
                        num_t = hm * wm
                        if offset + num_t <= len(a_img):
                            patch_scores = a_img[offset : offset + num_t].reshape(hm, wm)
                            row_means = patch_scores.mean(axis=1, keepdims=True)
                            a_img_smoothed[offset : offset + num_t] = np.repeat(row_means, wm, axis=1).flatten()
                            offset += num_t
                    
                    _save_step(step_idx, token_safe, a_img)
                    _save_step(step_idx, token_safe, a_img_smoothed, is_smoothed=True)

                    attention_img_all.append(a_img.copy())
                    if "accept" in token_text.lower() or "reject" in token_text.lower():
                        decision_img_all.append((step_idx, token_text, a_img.copy()))
                # Compute logits using a_img (image-only attended contribution) and compare with model logits.
                if a_img is not None and len(image_positions) > 0:
                    try:
                        # -- Part A: reconstruct image-only logits from a_img + V_img --
                        img_pos_t = torch.from_numpy(image_positions).long().to(device)
                        v_last = past_key_values[num_layers - 1][1]           # [1, kv_heads, kv_len, head_dim]
                        model_dtype = v_last.dtype
                        v_img = v_last[:, :, img_pos_t, :]                    # [1, kv_heads, n_img, head_dim]
                        groups = last_attn.num_heads // last_attn.num_key_value_heads
                        v_img_exp = v_img.repeat_interleave(groups, dim=1)    # [1, num_heads, n_img, head_dim]
                        
                        # Use headwise attention weights for reconstruction
                        # a_img_heads: [num_heads, n_img]
                        # V_img_exp[0]: [num_heads, n_img, head_dim]
                        attended_img = (a_img_heads[:, :, None] * v_img_exp[0]).sum(dim=1)  # [num_heads, head_dim]
                        attended_img = attended_img.reshape(1, 1, -1).to(model_dtype)         # [1, 1, hidden_dim]
                        with torch.no_grad():
                            with self.compute_loss_context_manager():
                                h_img = last_attn.o_proj(attended_img)        # [1, 1, hidden_dim]
                                if hasattr(model.model, "language_model"):
                                    norm_layer = model.model.language_model.norm
                                else:
                                    norm_layer = model.model.norm
                                norm_dtype = next(norm_layer.parameters()).dtype
                                head_dtype = next(model.lm_head.parameters()).dtype
                                h_norm = norm_layer(h_img.to(norm_dtype))
                                logits_from_a_img = model.lm_head(h_norm.to(head_dtype))[:, -1, :]  # [1, vocab]

                        # -- Part B: run forward with output_attentions=True to get model's actual logits --
                        _cache_pos_chk = torch.tensor(
                            [sample_inputs["input_ids"].shape[1] + step_idx], device=device
                        )
                        _attn_mask_chk = torch.cat(
                            [attention_mask, torch.ones((1, 1), dtype=attention_mask.dtype, device=device)],
                            dim=-1,
                        )
                        with torch.no_grad():
                            with self.compute_loss_context_manager():
                                _inputs_chk = model.prepare_inputs_for_generation(
                                    input_ids=torch.tensor([[cur_token_id]], device=device),
                                    past_key_values=past_key_values,
                                    attention_mask=_attn_mask_chk,
                                    use_cache=True,
                                    cache_position=_cache_pos_chk,
                                    pixel_values=sample_inputs.get("pixel_values"),
                                    image_grid_thw=sample_inputs.get("image_grid_thw"),
                                    pixel_values_videos=sample_inputs.get("pixel_values_videos"),
                                    video_grid_thw=sample_inputs.get("video_grid_thw"),
                                )
                                _out_chk = model(**_inputs_chk, output_attentions=True)
                        model_logits = _out_chk.logits[:, -1, :].float()

                        # -- Part C: extract model's image attention and compare with a_img --
                        if _out_chk.attentions is not None and _out_chk.attentions[-1] is not None:
                            _model_attn_last = _out_chk.attentions[-1]            # [1, heads, 1, kv_len]
                            _raw_img = _model_attn_last[0, :, -1, img_pos_t].mean(dim=0).float().cpu().numpy()
                            attn_max_diff = float(np.abs(a_img - _raw_img).max())
                            attn_corr = float(np.corrcoef(a_img, _raw_img)[0, 1]) if len(a_img) > 1 else float("nan")
                            attn_stats_str = f"max_diff={attn_max_diff:.4f} corr={attn_corr:.4f}"
                        else:
                            attn_stats_str = "max_diff=N/A corr=N/A"

                        top5_model = model_logits[0].topk(5).indices.tolist()
                        top5_a_img = logits_from_a_img[0].float().topk(5).indices.tolist()
                        top1_match = top5_model[0] == top5_a_img[0]
                        logger.info_rank0(
                            f"[attn_viz log] step={step_idx} "
                            f"a_img_vs_model_attn: {attn_stats_str} total_mass={a_img_sum:.4f} | "
                            f"logits top-1 match={top1_match} "
                            f"(model={self.processing_class.decode([top5_model[0]])!r} vs "
                            f"a_img={self.processing_class.decode([top5_a_img[0]])!r})"
                        )
                    except Exception as _exc:
                        import traceback
                        logger.warning_rank0(f"[attn_viz assert] step={step_idx} skipped: {traceback.format_exc()}")

                _save_step(step_idx, token_safe, a_img)

                if cur_token_id == self.processing_class.eos_token_id or "im_end" in token_text:
                    break

                # Update for next token
                cache_position = torch.tensor([sample_inputs["input_ids"].shape[1] + step_idx], device=device)
                attention_mask = torch.cat([attention_mask, torch.ones((1, 1), dtype=attention_mask.dtype, device=device)], dim=-1)

                # Forward for next token (flash attention, no output_attentions)
                with torch.no_grad():
                    with self.compute_loss_context_manager():
                        inputs_for_gen = model.prepare_inputs_for_generation(
                            input_ids=torch.tensor([[cur_token_id]], device=device),
                            past_key_values=past_key_values,
                            attention_mask=attention_mask,
                            use_cache=True,
                            cache_position=cache_position,
                            pixel_values=sample_inputs.get("pixel_values"),
                            image_grid_thw=sample_inputs.get("image_grid_thw"),
                            pixel_values_videos=sample_inputs.get("pixel_values_videos"),
                            video_grid_thw=sample_inputs.get("video_grid_thw"),
                        )
                        outputs = model(**inputs_for_gen)
                        next_token_id = _sample_next_token(outputs.logits[:, -1, :], generated_ids)
                        generated_ids.append(next_token_id)
                        past_key_values = outputs.past_key_values

            # Save summary numpy arrays and 2-D heatmap for image attention
            if attention_img_all:
                summary_img_dir = os.path.join(paper_dir, "summary_image")
                os.makedirs(summary_img_dir, exist_ok=True)
                
                # Section-wise attention stats
                blocks = self._get_block_metadata(paper_id)
                if blocks and decision_img_all:
                    sections = self._get_sections_from_blocks(blocks)
                    section_viz_data = []
                    for step_idx, token_text, a_img in decision_img_all:
                        weights = self._compute_section_weights_vision(a_img, image_sizes_merged, sections)
                        if weights:
                            section_viz_data.append({"step": step_idx, "token": token_text, "weights": weights})
                    
                    if section_viz_data:
                        with open(os.path.join(summary_img_dir, "section_attn.json"), "w") as f:
                            json.dump(section_viz_data, f, indent=2)
                        
                        # Plot the last decision token's section weights
                        last_data = section_viz_data[-1]
                        last_weights = last_data["weights"]
                        plt.figure(figsize=(10, 6))
                        names = list(last_weights.keys())
                        vals = list(last_weights.values())
                        plt.barh(names, vals)
                        plt.xlabel("Average Attention Weight")
                        plt.title(f"Section-wise Attention ({last_data['token']})")
                        plt.tight_layout()
                        plt.savefig(os.path.join(summary_img_dir, "section_attn_last.png"))
                        plt.close()
                        
                        # Semantic Heatmap: color the original PDF boxes
                        self._save_semantic_heatmaps(paper_id, last_weights, sections, image_paths, summary_img_dir)


                attn_img_2d = np.stack(attention_img_all)  # [steps, n_img_tokens]
                np.save(os.path.join(summary_img_dir, "attention_image_avg.npy"), attn_img_2d)

                # Save plots for decision tokens
                for (d_step, d_tok, d_a) in decision_img_all:
                    safe_d_tok = "".join(c for c in d_tok if c.isalnum() or c in ("_", "-")).strip() or "token"
                    plt.figure(figsize=(12, 4))
                    plt.plot(d_a)
                    plt.title("Mean Attention Distribution Across Image Patches")
                    plt.xlabel("Image Patch Index")
                    plt.ylabel("Mean Attention Score")
                    plt.tight_layout()
                    plt.savefig(os.path.join(summary_img_dir, f"attention_distribution_line_step{d_step}_{safe_d_tok}.png"), dpi=150)
                    plt.close()
                    
                    plt.figure(figsize=(8, 4))
                    plt.hist(d_a, bins=500, color='red', alpha=0.7)
                    plt.title(f"Histogram of Attention Scores for Decision Token '{d_tok}'")
                    plt.xlabel("Attention Score")
                    plt.ylabel("Frequency")
                    plt.xscale('log')
                    plt.yscale('log')
                    plt.tight_layout()
                    plt.savefig(os.path.join(summary_img_dir, f"attention_distribution_hist_step{d_step}_{safe_d_tok}.png"), dpi=150)
                    plt.close()

                # 2-D heatmap for image attention
                row_min = attn_img_2d.min(axis=1, keepdims=True)
                row_max = attn_img_2d.max(axis=1, keepdims=True)
                normed_2d_img = (attn_img_2d - row_min) / (row_max - row_min + 1e-8)
                SUMMARY_W = 1024
                SUMMARY_H = max(attn_img_2d.shape[0] * 8, 64)
                heatmap_2d_img = cv2.applyColorMap(
                    cv2.resize(np.uint8(255 * normed_2d_img), (SUMMARY_W, SUMMARY_H), interpolation=cv2.INTER_NEAREST),
                    cv2.COLORMAP_JET,
                )
                cv2.imwrite(os.path.join(summary_img_dir, "attention_image_avg_2d.jpg"), heatmap_2d_img)


            # Cleanup
            h_qproj.remove()
            h_pre.remove()
            q_cap.clear()
            all_generated_ids.append(list(generated_ids))
            del past_key_values, outputs, generated_ids
            torch.cuda.empty_cache()
            gc.collect()
            self._attn_viz_sample_idx += 1

        # Pad generated sequences to the same length and return as a tensor so that
        # save_predictions (which calls len(preds)) does not crash.
        pad_id = self.processing_class.pad_token_id
        max_gen_len = max((len(g) for g in all_generated_ids), default=1)
        padded = [
            g + [pad_id] * (max_gen_len - len(g)) for g in all_generated_ids
        ]
        generated_tokens = torch.tensor(padded, dtype=torch.long, device=device)
        return None, generated_tokens, inputs.get("labels")

    def _find_paper_token_range(
        self,
        prompt_ids_1d: "torch.Tensor",
        paper_start_marker: str = "\n\n",
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

        # Start the paper right after the marker (strip any leading whitespace in the marker itself)
        paper_start_char = marker_pos + len(paper_start_marker) - len(paper_start_marker.lstrip())

        # bisect_left gives the first token whose start char >= target
        paper_start_token = bisect.bisect_left(tok_char_starts, paper_start_char)
        paper_start_token = min(paper_start_token, len(prompt_ids_1d))

        return paper_start_token, len(prompt_ids_1d)

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
        all_generated_ids: list[list[int]] = []

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

            # Truncate to prompt-only
            labels_batch = inputs.get("labels")
            if labels_batch is not None and labels_batch.dim() == 2:
                labels_i = labels_batch[i]
                response_starts = (labels_i != IGNORE_INDEX).nonzero(as_tuple=True)[0]
                if len(response_starts) > 0:
                    prompt_end = response_starts[0].item()
                    if prompt_end > 0:
                        sample_inputs["input_ids"] = sample_inputs["input_ids"][:, :prompt_end]
                        if "attention_mask" in sample_inputs and sample_inputs["attention_mask"].dim() == 2:
                            sample_inputs["attention_mask"] = sample_inputs["attention_mask"][:, :prompt_end]
                        logger.info_rank0(f"[text-viz] Truncated to prompt-only: {labels_i.shape[0]} → {prompt_end} tokens")

            # Debug inputs and metadata
            logger.info_rank0(f"[text-viz] prediction_step keys: {list(inputs.keys())}")
            if "_metadata" in inputs:
                logger.info_rank0(f"[text-viz] Found _metadata: {inputs['_metadata']}")

            # Metadata extraction
            paper_id = None
            metadata = inputs.get("_metadata", [])
            if metadata and i < len(metadata) and metadata[i] is not None:
                if isinstance(metadata[i], dict):
                    paper_id = metadata[i].get("paper_id") or metadata[i].get("submission_id")
                elif isinstance(metadata[i], str):
                    try:
                        m_json = json.loads(metadata[i])
                        paper_id = m_json.get("paper_id") or m_json.get("submission_id")
                    except:
                        pass

            if paper_id is None:
                for k in ["submission_id", "paper_id", "id"]:
                    if k in inputs:
                        val = inputs[k][i]
                        paper_id = val if isinstance(val, str) else str(val)
                        break

            # Robust fallback: lookup directly from eval_dataset or data.json
            if paper_id is None:
                eval_ids = self._get_eval_paper_ids()
                if 0 <= self._attn_viz_sample_idx < len(eval_ids):
                    paper_id = eval_ids[self._attn_viz_sample_idx]

            if paper_id is None and hasattr(self, "eval_dataset"):
                try:
                    sample = self.eval_dataset[self._attn_viz_sample_idx]
                    m = sample.get("_metadata")
                    if m:
                        paper_id = m.get("submission_id") or m.get("paper_id")
                except:
                    pass

            if paper_id is None:
                logger.info_rank0(
                    f"[text-viz] No submission_id in metadata; using sequential ID"
                )
                paper_id = f"sample_{self._attn_viz_sample_idx:05d}"

            image_paths = []
            if "image_paths" in inputs:
                image_paths = inputs["image_paths"][i]
            elif "images" in inputs and isinstance(inputs["images"][i], (list, tuple)) and isinstance(inputs["images"][i][0], str):
                image_paths = inputs["images"][i]

            if not image_paths and hasattr(self, "eval_dataset"):
                try:
                    sample = self.eval_dataset[self._attn_viz_sample_idx]
                    image_paths = sample.get("images") or sample.get("image_paths") or []
                except:
                    pass

            if not image_paths and metadata and i < len(metadata) and metadata[i] is not None:
                if isinstance(metadata[i], dict):
                    image_paths = metadata[i].get("images") or metadata[i].get("image_paths") or []

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

            # Hook setup — only the last layer; no full attention matrix materialised
            last_layer = layers[num_layers - 1]
            last_attn  = last_layer.self_attn
            q_cap: dict = {}

            def _q_proj_hook(module, input, output):
                q_cap["q_raw"] = output.detach()

            def _self_attn_pre_hook(module, args, kwargs):
                pe = kwargs.get("position_embeddings")
                if pe is not None:
                    q_cap["cos"] = pe[0].detach()
                    q_cap["sin"] = pe[1].detach()

            h_qproj = last_attn.q_proj.register_forward_hook(_q_proj_hook)
            h_pre   = last_attn.register_forward_pre_hook(_self_attn_pre_hook, with_kwargs=True)

            paper_positions = np.arange(paper_start_token, paper_end_token)  # [num_paper_tokens]

            def _compute_a_text(past_key_values) -> Optional[np.ndarray]:
                if not q_cap or "q_raw" not in q_cap or "cos" not in q_cap:
                    return None
                if len(paper_positions) == 0:
                    return None
                try:
                    q_raw = q_cap["q_raw"]; cos = q_cap["cos"]; sin = q_cap["sin"]
                    bsz, q_len, _ = q_raw.shape
                    # Read head params from model config (works for both Qwen2 and Qwen2-VL)
                    cfg = model.config
                    num_heads   = getattr(cfg, "num_attention_heads", None) or getattr(last_attn, "num_heads", None)
                    num_kv_heads = getattr(cfg, "num_key_value_heads", None) or getattr(last_attn, "num_key_value_heads", num_heads)
                    head_dim    = getattr(cfg, "hidden_size", num_heads * 128) // num_heads
                    scaling     = head_dim ** -0.5
                    q = q_raw.view(bsz, q_len, num_heads, head_dim).transpose(1, 2)
                    # Use multimodal RoPE for VL models, standard RoPE for text models
                    mrope_section = None
                    rope_scaling = getattr(last_attn, "rope_scaling", None) or getattr(cfg, "rope_scaling", None)
                    if isinstance(rope_scaling, dict):
                        mrope_section = rope_scaling.get("mrope_section", None)
                    if mrope_section is not None:
                        from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import apply_multimodal_rotary_pos_emb
                        q_rot, _ = apply_multimodal_rotary_pos_emb(q, q, cos, sin, mrope_section)
                    else:
                        from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb
                        q_rot, _ = apply_rotary_pos_emb(q, q, cos, sin)
                    k_full = past_key_values[num_layers - 1][0]          # [1, kv_heads, kv_len, head_dim]
                    paper_pos_t = torch.from_numpy(paper_positions).long().to(k_full.device)
                    k_paper = k_full[:, :, paper_pos_t, :]               # [1, kv_heads, n_paper, head_dim]
                    groups = num_heads // num_kv_heads
                    k_paper = k_paper.repeat_interleave(groups, dim=1)   # [1, num_heads, n_paper, head_dim]
                    q_last = q_rot[:, :, -1:, :]                         # [1, num_heads, 1, head_dim]
                    attn_scores = torch.matmul(q_last, k_paper.transpose(-1, -2)) * scaling
                    attn_weights = torch.softmax(attn_scores.float(), dim=-1)
                    return attn_weights[0, :, 0, :].mean(dim=0).cpu().numpy()
                except Exception as exc:
                    logger.warning_rank0(f"_compute_a_text failed: {exc}")
                    return None

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

            # Maintain attention mask for generation loop manually
            generated_ids: list[int] = []
            attention_mask = sample_inputs.get("attention_mask")
            if attention_mask is None:
                attention_mask = torch.ones_like(sample_inputs["input_ids"])
            
            past_key_values = None
            cache_position = torch.arange(sample_inputs["input_ids"].shape[1], device=device)

            with torch.no_grad():
                with self.compute_loss_context_manager():
                    model_dtype = next(model.parameters()).dtype
                    for k, v in sample_inputs.items():
                        if isinstance(v, torch.Tensor) and torch.is_floating_point(v):
                            sample_inputs[k] = v.to(model_dtype)
                    
                    inputs_for_gen = model.prepare_inputs_for_generation(
                        **sample_inputs,
                        past_key_values=past_key_values,
                        use_cache=True,
                        cache_position=cache_position,
                    )

                    outputs = model(**inputs_for_gen)
                    next_token_id = _sample_next_token(outputs.logits[:, -1, :], generated_ids)
                    generated_ids.append(next_token_id)
                    past_key_values = outputs.past_key_values

            # Accumulators for the 2-D summary
            attention_avg_all: list[np.ndarray] = []   # per step: avg across layers
            attention_last_all: list[np.ndarray] = []  # per step: last layer only
            decision_text_all: list[tuple[int, str, np.ndarray]] = []

            # Precompute output dirs to avoid repeated joins
            def _get_out_dirs(variant=""):
                avg_d = os.path.join(paper_dir, "average", f"images{variant}")
                last_d = os.path.join(paper_dir, "last", f"images{variant}")
                os.makedirs(avg_d, exist_ok=True)
                os.makedirs(last_d, exist_ok=True)
                return avg_d, last_d

            dirs_map = {
                "": _get_out_dirs(""),
                "_top5": _get_out_dirs("_top5"),
                "_top10": _get_out_dirs("_top10"),
                "_top25": _get_out_dirs("_top25"),
                "_smoothed": _get_out_dirs("_smoothed"),
                "_top5_smoothed": _get_out_dirs("_top5_smoothed"),
                "_top10_smoothed": _get_out_dirs("_top10_smoothed"),
                "_top25_smoothed": _get_out_dirs("_top25_smoothed"),
            }
            HEATMAP_W = min(num_paper_tokens, 1024)

            # Token-by-token generation
            for step_idx in range(max_new_tokens):
                cur_token_id = generated_ids[-1]
                token_text = self.processing_class.decode([cur_token_id]).strip() or f"token_{cur_token_id}"
                token_safe = (
                    "".join(c for c in token_text if c.isalnum() or c in ("_", "-")).strip()
                    or f"token_{cur_token_id}"
                )

                a_text = _compute_a_text(past_key_values)

                if a_text is not None:
                    # Smoothing: average over every 200 tokens
                    a_text_smoothed = np.zeros_like(a_text)
                    chunk_size = 200
                    for i in range(0, len(a_text), chunk_size):
                        a_text_smoothed[i:i+chunk_size] = a_text[i:i+chunk_size].mean()
                    a_text = a_text_smoothed

                    attention_avg_all.append(a_text.copy())
                    attention_last_all.append(a_text.copy())   # same — single layer captured
                    if "accept" in token_text.lower() or "reject" in token_text.lower():
                        decision_text_all.append((step_idx, token_text, a_text.copy()))

                    normed = sum_normalise(a_text)
                    
                    def _save_bar(normed_arr, variant_name):
                        bar = cv2.applyColorMap(
                            cv2.resize(np.uint8(255 * normed_arr).reshape(1, -1), (HEATMAP_W, 64),
                                       interpolation=cv2.INTER_AREA),
                            cv2.COLORMAP_JET,
                        )
                        ad, ld = dirs_map[variant_name]
                        cv2.imwrite(os.path.join(ad, f"step_{step_idx:03d}_{token_safe}.jpg"), bar)
                        cv2.imwrite(os.path.join(ld, f"step_{step_idx:03d}_{token_safe}.jpg"), bar)

                    def _save_variants(arr, suffix=""):
                        n_arr = sum_normalise(arr)
                        _save_bar(n_arr, f"{suffix}")
                        for k in [5, 10, 25]:
                            threshold = np.percentile(n_arr, 100 - k)
                            filtered = np.where(n_arr >= threshold, n_arr, 0.0)
                            _save_bar(sum_normalise(filtered), f"_top{k}{suffix}")

                    _save_variants(a_text)
                    _save_variants(a_text_smoothed, suffix="_smoothed")

                if cur_token_id == self.processing_class.eos_token_id or "im_end" in token_text:
                    break

                # Update for next token
                cache_position = torch.tensor([sample_inputs["input_ids"].shape[1] + step_idx], device=device)
                attention_mask = torch.cat([attention_mask, torch.ones((1, 1), dtype=attention_mask.dtype, device=device)], dim=-1)

                # Forward for next token
                with torch.no_grad():
                    with self.compute_loss_context_manager():
                        inputs_for_gen = model.prepare_inputs_for_generation(
                            input_ids=torch.tensor([[cur_token_id]], device=device),
                            past_key_values=past_key_values,
                            attention_mask=attention_mask,
                            use_cache=True,
                            cache_position=cache_position,
                            pixel_values=sample_inputs.get("pixel_values"),
                            image_grid_thw=sample_inputs.get("image_grid_thw"),
                            pixel_values_videos=sample_inputs.get("pixel_values_videos"),
                            video_grid_thw=sample_inputs.get("video_grid_thw"),
                        )
                        outputs = model(**inputs_for_gen)
                        next_token_id = _sample_next_token(outputs.logits[:, -1, :], generated_ids)
                        generated_ids.append(next_token_id)
                        past_key_values = outputs.past_key_values

            # Save summary numpy arrays and 2-D heatmap
            if attention_avg_all:
                summary_dir = os.path.join(paper_dir, "summary")
                os.makedirs(summary_dir, exist_ok=True)

                attn_avg_2d = np.stack(attention_avg_all)   # [steps, paper_tokens]
                np.save(os.path.join(summary_dir, "attention_avg.npy"), attn_avg_2d)

                # Save plots for decision tokens
                for (d_step, d_tok, d_a) in decision_text_all:
                    safe_d_tok = "".join(c for c in d_tok if c.isalnum() or c in ("_", "-")).strip() or "token"
                    plt.figure(figsize=(12, 4))
                    plt.plot(d_a)
                    plt.title("Mean Attention Distribution Across Paper Tokens")
                    plt.xlabel("Paper Token Index")
                    plt.ylabel("Mean Attention Score")
                    plt.tight_layout()
                    plt.savefig(os.path.join(summary_dir, f"attention_distribution_line_step{d_step}_{safe_d_tok}.png"), dpi=150)
                    plt.close()
                    
                    plt.figure(figsize=(8, 4))
                    plt.hist(d_a, bins=500, color='red', alpha=0.7)
                    plt.title(f"Histogram of Attention Scores for Decision Token '{d_tok}'")
                    plt.xscale('log')
                    plt.yscale('log')
                    plt.xlabel("Attention Score")
                    plt.ylabel("Frequency")
                    plt.tight_layout()
                    plt.savefig(os.path.join(summary_dir, f"attention_distribution_hist_step{d_step}_{safe_d_tok}.png"), dpi=150)
                    plt.close()
                
                # Section-wise attention stats (Text)
                blocks = self._get_block_metadata(paper_id)
                if blocks and decision_text_all:
                    sections = self._get_sections_from_blocks(blocks)
                    paper_token_ids = prompt_ids[paper_start_token:paper_end_token].tolist()
                    section_viz_data_text = []
                    for d_step, d_tok, d_a in decision_text_all:
                        weights = self._compute_section_weights_text(d_a, paper_token_ids, sections)
                        if weights:
                            section_viz_data_text.append({"step": d_step, "token": d_tok, "weights": weights})
                    
                    if section_viz_data_text:
                        with open(os.path.join(summary_dir, "section_attn.json"), "w") as f:
                            json.dump(section_viz_data_text, f, indent=2)
                        
                        # Plot the last decision token's section weights
                        last_data = section_viz_data_text[-1]
                        last_weights = last_data["weights"]
                        plt.figure(figsize=(10, 6))
                        names = list(last_weights.keys())
                        vals = list(last_weights.values())
                        plt.barh(names, vals)
                        plt.xlabel("Average Attention Weight")
                        plt.title(f"Section-wise Attention ({last_data['token']})")
                        plt.tight_layout()
                        plt.savefig(os.path.join(summary_dir, "section_attn_last.png"))
                        plt.close()
                        
                        # Semantic Heatmap: color the original PDF boxes
                        self._save_semantic_heatmaps(paper_id, last_weights, sections, image_paths, summary_dir)

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

                # ── Interactive HTML ──────────────────────────────────────────
                try:
                    n_steps = attn_avg_2d.shape[0]
                    # Paper token texts (one per paper position)
                    _paper_toks = [
                        self.processing_class.decode(
                            [prompt_ids[paper_start_token + j].item()],
                            skip_special_tokens=False,
                        )
                        for j in range(num_paper_tokens)
                    ]
                    # Generated token texts aligned with attention steps
                    _gen_toks = [
                        self.processing_class.decode([generated_ids[j]], skip_special_tokens=False)
                        for j in range(min(n_steps, len(generated_ids)))
                    ]
                    
                    # Section mapping
                    _section_mapping = ["None"] * num_paper_tokens
                    if blocks:
                        _section_mapping = self._get_token_to_section_mapping(paper_token_ids, sections)
                    
                    _tok_js  = json.dumps(_paper_toks)
                    _sec_js  = json.dumps(_section_mapping)
                    _opts = "\n".join(
                        '<option value="{}">[{}] {}</option>'.format(
                            i, i,
                            t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"),
                        )
                        for i, t in enumerate(_gen_toks)
                    )
                    
                    def _write_html(arr_2d, variant=""):
                        _attn_js = json.dumps(arr_2d.round(4).tolist())
                        _html = (
                            "<!DOCTYPE html>\n<html lang=\"en\">\n<head><meta charset=\"UTF-8\">"
                            "<title>Attention{v}: {pid}</title>\n<style>\n"
                            "body{{font-family:Georgia,serif;line-height:1.9;padding:20px 40px;"
                            "max-width:1100px;margin:auto}}\n"
                            ".ctrl{{position:sticky;top:0;background:#fff;padding:8px 0;"
                            "border-bottom:2px solid #ddd;margin-bottom:10px;z-index:10}}\n"
                            "select{{font-size:14px;padding:4px;margin-left:6px}}\n"
                            ".scale{{display:flex;align-items:center;gap:8px;margin-top:8px;font-size:12px;color:#444}}\n"
                            ".scale-bar{{flex:1;height:14px;border-radius:3px;"
                            "background:linear-gradient(to right,rgba(220,50,30,0),rgba(220,50,30,1));"
                            "border:1px solid #ccc}}\n"
                            ".paper{{white-space:pre-wrap;font-size:14px;line-height:2}}\n"
                            ".sec-info{{font-size:12px;color:#666;margin-top:4px;height:1.2em}}\n"
                            "</style></head>\n<body>\n"
                            "<h2>Text Attention{v} &mdash; {pid}</h2>\n"
                            "<div class=\"ctrl\">"
                            "<label><b>Generated token:</b>"
                            "<select id=\"sel\" onchange=\"render(+this.value)\">\n{opts}\n"
                            "</select></label>\n"
                            "<div class=\"sec-info\" id=\"sec-info\">Section: -</div>\n"
                            "<div class=\"scale\">"
                            "<span id=\"lo-lbl\">0.0000</span>"
                            "<div class=\"scale-bar\"></div>"
                            "<span id=\"hi-lbl\">1.0000</span>"
                            "</div></div>\n"
                            "<div class=\"paper\" id=\"paper\"></div>\n<script>\n"
                            "const A={attn};\nconst toks={toks};\nconst secs={secs};\n"
                            "const uniqueSecs = Array.from(new Set(secs));\n"
                            "function esc(s){{return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');}}\n"
                            "function render(step){{\n"
                            "  const a=A[step];\n"
                            "  let lo=a[0],hi=a[0];\n"
                            "  for(const v of a){{if(v<lo)lo=v;if(v>hi)hi=v;}}\n"
                            "  document.getElementById('lo-lbl').textContent=lo.toFixed(5);\n"
                            "  document.getElementById('hi-lbl').textContent=hi.toFixed(5);\n"
                            "  const rng=hi-lo+1e-9;\n"
                            "  document.getElementById('paper').innerHTML=toks.map((t,i)=>{{\n"
                            "    const norm=(a[i]-lo)/rng;\n"
                            "    const s=secs[i];\n"
                            "    const sIdx = uniqueSecs.indexOf(s);\n"
                            "    const border = (s==='None' || s==='Preamble') ? 'none' : '2px solid hsl('+(sIdx*137 % 360)+',60%,50%)';\n"
                            "    return '<span style=\"background:rgba(220,50,30,'+norm.toFixed(3)+'); border-bottom:'+border+'\""
                            " onmouseover=\"document.getElementById(\\'sec-info\\').textContent=\\'Section: \\'+secs[i]\""
                            " title=\"['+secs[i]+'] '+a[i].toFixed(5)+'\">'+esc(t)+'</span>';\n"
                            "  }}).join('');\n}}\n"
                            "render(0);\n</script>\n</body>\n</html>\n"
                        ).format(pid=paper_id, opts=_opts, attn=_attn_js, toks=_tok_js, secs=_sec_js, v=variant)
                        html_path = os.path.join(paper_dir, f"attention{variant}.html")
                        with open(html_path, "w", encoding="utf-8") as _f:
                            _f.write(_html)
                        
                    _write_html(attn_avg_2d, "")
                    for k in [5, 10, 25]:
                        th = np.percentile(attn_avg_2d, 100 - k, axis=1, keepdims=True)
                        filtered_2d = np.where(attn_avg_2d >= th, attn_avg_2d, 0.0)
                        _write_html(filtered_2d, f"_top{k}")

                    logger.info_rank0(f"[text-viz] Wrote attention HTMLs for {paper_id}")
                except Exception as _html_exc:
                    logger.warning_rank0(f"[text-viz] HTML generation failed: {_html_exc}")

            # Cleanup
            h_qproj.remove()
            h_pre.remove()
            q_cap.clear()
            all_generated_ids.append(list(generated_ids))
            del past_key_values, outputs, generated_ids
            torch.cuda.empty_cache()
            gc.collect()
            self._attn_viz_sample_idx += 1

        pad_id = self.processing_class.pad_token_id
        max_gen_len = max((len(g) for g in all_generated_ids), default=1)
        padded = [
            g + [pad_id] * (max_gen_len - len(g)) for g in all_generated_ids
        ]
        generated_tokens = torch.tensor(padded, dtype=torch.long, device=device)
        return None, generated_tokens, inputs.get("labels")

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
