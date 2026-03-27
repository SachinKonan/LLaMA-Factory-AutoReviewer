## Overview

First RL training run for the NoArXiv acceptance prediction task. Trained **Qwen3-4B-Thinking** with **GRPO** (Group Relative Policy Optimization) on the `review_roles` prompt selected from the prompt ablation study (Task 1).

**Training setup:**
- **Algorithm**: GRPO with KL penalty (coef=0.001)
- **Batch size**: 256, mini-batch 128, 10 samples/prompt
- **Learning rate**: 1e-6 with 4 warmup steps, max grad norm 0.5
- **Max turns**: 1 (text-only, no search)
- **Max generate length**: 4096
- **Framework**: SkyRL with vLLM generator, FSDP2 strategy, Liger kernel
- **GPUs**: 8x H100 (PLI partition)

**Training ran for ~49 global steps** with eval dumps at steps 20, 40, and 49.

## Results & Status

<table fit-page-width="true" header-row="true">
<tr>
<td>Run</td>
<td>Status</td>
<td>Job ID</td>
<td>Steps</td>
<td>PROOF</td>
</tr>
<tr>
<td>Initial run</td>
<td><span color="green">Done</span></td>
<td>4680730</td>
<td>~20</td>
<td>`exports/noarxiv_qwen3_4b_review_roles/4680730/`</td>
</tr>
<tr>
<td>Continued run</td>
<td><span color="green">Done</span></td>
<td>4738814</td>
<td>20-49</td>
<td>`exports/noarxiv_qwen3_4b_review_roles/4738814/`</td>
</tr>
<tr>
<td>Eval @ step 20</td>
<td><span color="green">Done</span></td>
<td>—</td>
<td>20</td>
<td>`exports/.../4738814/exports/dumped_evals/step_20/`</td>
</tr>
<tr>
<td>Eval @ step 40</td>
<td><span color="green">Done</span></td>
<td>—</td>
<td>40</td>
<td>`exports/.../4738814/exports/dumped_evals/step_40/`</td>
</tr>
<tr>
<td>Eval @ step 49</td>
<td><span color="green">Done</span></td>
<td>—</td>
<td>49</td>
<td>`exports/.../4738814/exports/dumped_evals/step_49/`</td>
</tr>
<tr>
<td>Post-RL eval (passat)</td>
<td><span color="green">Done</span></td>
<td>—</td>
<td>20</td>
<td>`results/passat/review_roles_qwen3_4b_20stepRL.jsonl`</td>
</tr>
</table>

## Figures

<image source="https://raw.githubusercontent.com/SachinKonan/SkyRL/searchenvopt/skyrl-train/results/passat/prompt_ablation_run2.png">NoArXiv Prompt Ablation (Run 2) — shows before/after RL improvement for review\_roles on Qwen3-4B</image>

## WandB Logs

Training metrics logged via Weights & Biases (offline mode).

- **Project**: `noarxiv_acceptance`
- **Run ID**: `802kgznq` (continued run, job 4738814)
- **Offline logs**: `wandb/offline-run-*-802kgznq/`
- **Sync status**: Synced to cloud
- **Initial run**: job 4680730, run ID `jtp6e5q1` (approximate)
- **Key metrics**: policy\_loss, policy\_entropy, policy\_kl, avg\_raw\_reward, avg\_pass\_at\_10

## Key Files

- **Training sbatch**: `examples/search/sbatch/run_noarxiv_qwen3_4b_review_roles_pli.sbatch`
- **Eval conversion**: `scripts/convert_dumped_evals_to_passat.py`
- **Plotting**: `scripts/plot_pass_at_k.py`
- **Training logs**: `logs/noarxiv/review_roles/pli/4738814.{out,err}`
- **Exports**: `exports/noarxiv_qwen3_4b_review_roles/4738814/exports/`
- **Post-RL results**: `results/passat/review_roles_qwen3_4b_20stepRL.jsonl` (397 MB)
- **Dataset**: `noarxiv_review_roles_iclr` (ICLR balanced binary)

## Conclusion

- RL training with GRPO on the `review_roles` prompt **improves all metrics** compared to the base Qwen3-4B model.
- The updated prompt ablation figure (Run 2) shows clear before/after gains — the 20-step RL checkpoint outperforms the base model on pass@k, majority@k, and threshold metrics.
- Training converged within ~49 steps (1 epoch over the ICLR dataset with bz=256).
- **Next step**: Try GSPO loss with asymmetric clip higher to see if further gains are possible (Task 3).

▶ Sbatch Configuration
	**Job name**: `noarxiv_qwen3_4b_review_roles`
	**GPUs**: 8x H100 (PLI partition)
	**Memory**: 300G, 20 CPUs
	**Time**: 36:00:00
	**Model**: Qwen3-4B-Thinking
	**Dataset**: `noarxiv_review_roles_iclr`
	**train\_batch\_size**: 256
	**policy\_mini\_batch\_size**: 128
	**n\_samples\_per\_prompt**: 10
	**learning\_rate**: 1e-6
	**KL coeff**: 0.001
	**max\_grad\_norm**: 0.5
	**max\_turns**: 1
	**max\_generate\_length**: 4096
	**eval\_interval**: 20
	**save\_interval**: 5
	**strategy**: fsdp2
	**Liger kernel**: enabled
	**advantage\_estimator**: grpo
