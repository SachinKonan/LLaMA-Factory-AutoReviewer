## Overview

RL training for the **ArXiv** (search-augmented) acceptance prediction task using **GSPO loss + asymmetric clip higher** on the `one_search` prompt selected from the ArXiv prompt ablation (Task 4).

This is the search-augmented counterpart to Task 3 (NoArXiv GSPO). The model learns to:
1. **Turn 1**: Generate a search query to find relevant arXiv papers
2. **Turn 2**: Read the retrieved papers and make an accept/reject decision

**Key setup:**
- **Algorithm**: GRPO + GSPO loss + eps\_clip\_high=0.28
- **Loss reduction**: `sequence_mean`
- **Max turns**: 2 (search + decide)
- **Max generate length**: 1000
- **Retrieval**: Qwen3 embeddings + FAISS Flat index, topk=10
- **Dataset**: `arxiv_iclr_balanced_one_search` (year-balanced, one\_search system prompt)
- **GPUs**: 8x H100 (PLI partition) — 1 GPU reserved for retrieval server

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
<td><span color="orange">Pending</span></td>
<td>4812215</td>
<td>`run_arxiv_qwen3_4b_one_search_gspo_cliphigh_pli.sbatch`</td>
<td>—</td>
</tr>
</table>

## WandB Logs

Training metrics logged via Weights & Biases (offline mode).

- **Project**: `arxiv_acceptance`
- **Run ID**: Pending (job 4812215 not yet started)
- **Offline logs**: `wandb/offline-run-*-{run_id}/` (will be available after training starts)
- **Sync status**: Not synced
- **Key metrics**: policy\_loss, policy\_entropy, policy\_kl, avg\_raw\_reward, avg\_pass\_at\_10

## Key Files

- **Sbatch**: `examples/search/sbatch/run_arxiv_qwen3_4b_one_search_gspo_cliphigh_pli.sbatch`
- **Dataset script**: `examples/search/searchr1_arxiv_dataset.py`
- **Exports (when done)**: `exports/arxiv_qwen3_4b_one_search_gspo_cliphigh/{JOB_ID}/`
- **Logs (when done)**: `logs/arxiv/one_search_gspo_cliphigh/pli/{JOB_ID}.{out,err}`
- **Retrieval index**: `data/searchr1_original/arxiv/qwen3_06_embed/qwen3_Flat.index`
- **Corpus**: `data/searchr1_original/arxiv/arxiv_wikiformat.jsonl`

## Conclusion

*In progress — update when training completes with eval results and comparison to base model and NoArXiv Task 3.*

▶ Sbatch Configuration
	**Job name**: `arxiv_qwen3_4b_one_search_gspo_cliphigh`
	**GPUs**: 8x H100 (PLI partition)
	**Memory**: 300G, 20 CPUs
	**Time**: 36:00:00
	**Model**: Qwen3-4B-Thinking
	**Dataset**: `arxiv_iclr_balanced_one_search`
	**train\_batch\_size**: 256
	**policy\_mini\_batch\_size**: 128
	**n\_samples\_per\_prompt**: 10
	**learning\_rate**: 1e-6
	**KL coeff**: 0.001
	**max\_grad\_norm**: 0.5
	**max\_turns**: 2
	**max\_generate\_length**: 1000
	**eval\_interval**: 20
	**save\_interval**: 5
	**strategy**: fsdp2
	**Liger kernel**: enabled
	**advantage\_estimator**: grpo
	**policy\_loss\_type**: gspo
	**loss\_reduction**: sequence\_mean
	**eps\_clip\_high**: 0.28
	**Retrieval**: Qwen3 embeddings, FAISS Flat index, topk=10
	**Search URL**: `http://127.0.0.1:8000/retrieve`
	**Stop tokens**: `\</ssearch\>`, `\</asearch\>`, `\</answer\>`
