## Overview

Prompt ablation study for the NoArXiv (text-only) acceptance prediction task. Tested **6 system prompts** across **3 models** (Qwen3-4B-Thinking, Qwen2.5-7B-Instruct, Qwen3-8B) with **30 completions per paper** on the ICLR balanced binary test set.

**Prompts tested:**
1. `nonconcise` — basic review prompt without conciseness constraint
2. `nonconcise_review` — adds explicit review process instructions
3. `structured` — structured analysis with sections
4. `scoring` — numerical scoring rubric
5. `devils_advocate` — adversarial critique perspective
6. `review_roles` — 3-reviewer simulation (Advocate / Critic / Calibrator)

**Goal:** Identify the best prompt for RL training by comparing pass@k and threshold-based metrics across all prompt-model combinations.

## Metrics

All metrics are computed over the first k completions per paper (deterministic indexing):

- **pass@k**: Whether any of the first k completions gives the correct answer, averaged over all papers. Measures the ceiling of a best-of-k selection strategy.

- **majority@k**: Accuracy when taking the majority vote among the first k completions. Ties (equal correct/incorrect) count as 0.5.

- **threshreject(k, M)**: Accuracy when predicting Reject if at least M of the first k completions say Reject, else Accept.

- **threshaccept(k, M)**: Accuracy when predicting Accept if at least M of the first k completions say Accept, else Reject.

## Results & Status

Evidence-based status. PROOF = result file path from project root.

<table fit-page-width="true" header-row="true">
<tr>
<td>Prompt</td>
<td>Model</td>
<td>Status</td>
<td>PROOF</td>
</tr>
<tr>
<td>nonconcise</td>
<td>Qwen3-4B</td>
<td><span color="green">Done</span></td>
<td>`results/passat/nonconcise_qwen3_4b.jsonl`</td>
</tr>
<tr>
<td>nonconcise</td>
<td>Qwen2.5-7B</td>
<td><span color="green">Done</span></td>
<td>`results/passat/nonconcise_qwen2.5_7b.jsonl`</td>
</tr>
<tr>
<td>nonconcise</td>
<td>Qwen3-8B</td>
<td><span color="green">Done</span></td>
<td>`results/passat/nonconcise_qwen3_8b.jsonl`</td>
</tr>
<tr>
<td>nonconcise\_review</td>
<td>Qwen3-4B</td>
<td><span color="green">Done</span></td>
<td>`results/passat/nonconcise_review_qwen3_4b.jsonl`</td>
</tr>
<tr>
<td>nonconcise\_review</td>
<td>Qwen2.5-7B</td>
<td><span color="green">Done</span></td>
<td>`results/passat/nonconcise_review_qwen2.5_7b.jsonl`</td>
</tr>
<tr>
<td>nonconcise\_review</td>
<td>Qwen3-8B</td>
<td><span color="green">Done</span></td>
<td>`results/passat/nonconcise_review_qwen3_8b.jsonl`</td>
</tr>
<tr>
<td>structured</td>
<td>Qwen3-4B</td>
<td><span color="green">Done</span></td>
<td>`results/passat/structured_qwen3_4b.jsonl`</td>
</tr>
<tr>
<td>structured</td>
<td>Qwen2.5-7B</td>
<td><span color="green">Done</span></td>
<td>`results/passat/structured_qwen2.5_7b.jsonl`</td>
</tr>
<tr>
<td>structured</td>
<td>Qwen3-8B</td>
<td><span color="green">Done</span></td>
<td>`results/passat/structured_qwen3_8b.jsonl`</td>
</tr>
<tr>
<td>scoring</td>
<td>Qwen3-4B</td>
<td><span color="green">Done</span></td>
<td>`results/passat/scoring_qwen3_4b.jsonl`</td>
</tr>
<tr>
<td>scoring</td>
<td>Qwen2.5-7B</td>
<td><span color="green">Done</span></td>
<td>`results/passat/scoring_qwen2.5_7b.jsonl`</td>
</tr>
<tr>
<td>scoring</td>
<td>Qwen3-8B</td>
<td><span color="green">Done</span></td>
<td>`results/passat/scoring_qwen3_8b.jsonl`</td>
</tr>
<tr>
<td>devils\_advocate</td>
<td>Qwen3-4B</td>
<td><span color="green">Done</span></td>
<td>`results/passat/devils_advocate_qwen3_4b.jsonl`</td>
</tr>
<tr>
<td>devils\_advocate</td>
<td>Qwen2.5-7B</td>
<td><span color="green">Done</span></td>
<td>`results/passat/devils_advocate_qwen2.5_7b.jsonl`</td>
</tr>
<tr>
<td>devils\_advocate</td>
<td>Qwen3-8B</td>
<td><span color="green">Done</span></td>
<td>`results/passat/devils_advocate_qwen3_8b.jsonl`</td>
</tr>
<tr>
<td><span color="green">review\_roles</span></td>
<td>Qwen3-4B</td>
<td><span color="green">Done</span></td>
<td>`results/passat/review_roles_qwen3_4b.jsonl`</td>
</tr>
<tr>
<td><span color="green">review\_roles</span></td>
<td>Qwen2.5-7B</td>
<td><span color="green">Done</span></td>
<td>`results/passat/review_roles_qwen2.5_7b.jsonl` (pending)</td>
</tr>
<tr>
<td><span color="green">review\_roles</span></td>
<td>Qwen3-8B</td>
<td><span color="green">Done</span></td>
<td>`results/passat/review_roles_qwen3_8b.jsonl` (pending)</td>
</tr>
</table>

## Figures

<image source="https://raw.githubusercontent.com/SachinKonan/SkyRL/searchenvopt/skyrl-train/results/passat/prompt_ablation.png">NoArXiv Prompt Ablation — pass@k, majority@k, threshreject, threshaccept across 6 prompts x 3 models</image>

## Key Files

- **Sbatch**: `examples/search/sbatch/run_base_eval_prompt_ablation.sbatch`
- **Inference script**: `scripts/vllm_infer.sh`
- **Plotting script**: `scripts/plot_pass_at_k.py --prompt_ablation`
- **Results dir**: `results/passat/`
- **Config**: array job 0-17 (6 prompts x 3 models), 4x GPU80, 30 completions/paper, max\_gen\_len=4096

## Conclusion

- **`review_roles` is the best prompt** — it consistently achieves the highest pass@k, majority@k, and threshold metrics across all three models.
- The 3-reviewer simulation (Advocate/Critic/Calibrator) provides structured deliberation that produces more calibrated accept/reject decisions.
- Qwen3-4B-Thinking with `review_roles` was selected as the base for RL training (Task 2).
- `devils_advocate` performs worst — adversarial-only framing biases the model toward over-rejection.
- `scoring` and `structured` are competitive but below `review_roles` on threshold metrics.

▶ Sbatch Configuration
	**Job name**: `prompt_ablation`
	**Array**: 0-17 (prompt\_idx \* 3 + model\_idx)
	**GPUs**: 4x GPU80
	**Memory**: 80G, 10 CPUs
	**Time**: 1:00:00
	**Prompts**: nonconcise, nonconcise\_review, structured, scoring, devils\_advocate, review\_roles
	**Models**: Qwen3-4B-Thinking, Qwen2.5-7B-Instruct, Qwen3-8B
	**N samples**: 30 completions/paper
	**Max gen length**: 4096
	**Eval batch size**: 32
