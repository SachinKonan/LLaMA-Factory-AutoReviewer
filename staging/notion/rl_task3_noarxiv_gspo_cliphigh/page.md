## Overview

Follow-up RL training experiment using **GSPO loss** (sequence-level importance sampling clipping) with **asymmetric clip higher** (`eps_clip_high=0.28`, DAPO-style) on the `review_roles` prompt. This replaces the standard PPO/GRPO token-level clipping with sequence-level clipping to better handle long-form text generation.

**Key differences from Task 2 (standard GRPO):**
- **Policy loss**: GSPO instead of standard GRPO token-level clipping
- **Loss reduction**: `sequence_mean` (recommended for GSPO)
- **eps\_clip\_high**: 0.28 (asymmetric — allows larger positive ratio updates)
- All other hyperparameters identical to Task 2

**Hypothesis:** GSPO's sequence-level IS clipping is better suited for long-form review generation where token-level clipping can be too conservative. The asymmetric clip higher allows the model to more aggressively reinforce high-reward completions.

## Results & Status

<table fit-page-width="true" header-row="true">
<tr>
<td>Variant</td>
<td>Status</td>
<td>Job ID</td>
<td>Sbatch</td>
<td>PROOF</td>
</tr>
<tr>
<td>GSPO + clip\_high=0.28</td>
<td><span color="blue">In progress</span></td>
<td>4760696</td>
<td>`run_noarxiv_qwen3_4b_review_roles_gspo_cliphigh_pli.sbatch`</td>
<td>`exports/noarxiv_qwen3_4b_review_roles_gspo_cliphigh/4760696/`</td>
</tr>
<tr>
<td>GSPO (no clip\_high)</td>
<td><span color="orange">Pending</span></td>
<td>4812226</td>
<td>`run_noarxiv_qwen3_4b_review_roles_gspo_pli.sbatch`</td>
<td>—</td>
</tr>
</table>

## WandB Logs

Training metrics logged via Weights & Biases (offline mode).

- **Project**: `noarxiv_acceptance`
- **Run ID**: `xagrjcv5` (GSPO + clip\_high, job 4760696)
- **Offline logs**: `wandb/offline-run-*-xagrjcv5/`
- **Sync status**: Not synced
- **Key metrics**: policy\_loss, policy\_entropy, policy\_kl, avg\_raw\_reward, avg\_pass\_at\_10

## Key Files

- **Sbatch (clip higher)**: `examples/search/sbatch/run_noarxiv_qwen3_4b_review_roles_gspo_cliphigh_pli.sbatch`
- **Sbatch (no clip higher)**: `examples/search/sbatch/run_noarxiv_qwen3_4b_review_roles_gspo_pli.sbatch`
- **Exports**: `exports/noarxiv_qwen3_4b_review_roles_gspo_cliphigh/4760696/`
- **Logs**: `logs/noarxiv/review_roles_gspo_cliphigh/pli/4760696.{out,err}`
- **Dataset**: `noarxiv_review_roles_iclr`

## Conclusion

*In progress — update when training completes with eval results and comparison to Task 2 (standard GRPO).*

▶ Sbatch Configuration
	**Job name**: `noarxiv_qwen3_4b_review_roles_gspo_cliphigh`
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
	**policy\_loss\_type**: gspo
	**loss\_reduction**: sequence\_mean
	**eps\_clip\_high**: 0.28
