## Overview

Prompt ablation study for the **ArXiv** (search-augmented) acceptance prediction task. Tested **7 search-augmented system prompts** on **Qwen3-4B-Thinking** with **topk={5, 10}** retrieved papers, yielding 14 configurations. Each configuration ran with 30 completions per paper on the ICLR balanced binary test set.

Unlike the NoArXiv ablation, these prompts involve **multi-turn search interactions** where the model can query an arXiv retrieval server (Qwen3 embeddings + FAISS index) to find relevant prior work before making its accept/reject decision.

**Prompts tested:**
1. `basic` — minimal search prompt, 4 turns
2. `one_search` — single search query, 2 turns (search then decide)
3. `reviewer_turns` — iterative reviewer perspective across 4 turns
4. `claim_contrast` — claim-based contrastive search, 4 turns
5. `prior_art` — prior art comparison focus, 4 turns
6. `debate_search` — debate-style with search evidence, 4 turns
7. `structured_evidence` — structured evidence gathering, 4 turns

## Metrics

All metrics are computed over the first k completions per paper (deterministic indexing):

- **pass@k**: Whether any of the first k completions gives the correct answer, averaged over all papers. Measures the ceiling of a best-of-k selection strategy.

- **majority@k**: Accuracy when taking the majority vote among the first k completions. Ties (equal correct/incorrect) count as 0.5.

- **threshreject(k, M)**: Accuracy when predicting Reject if at least M of the first k completions say Reject, else Accept.

- **threshaccept(k, M)**: Accuracy when predicting Accept if at least M of the first k completions say Accept, else Reject.

## Results & Status

<table fit-page-width="true" header-row="true">
<tr>
<td>Prompt</td>
<td>TopK</td>
<td>Status</td>
<td>PROOF</td>
</tr>
<tr>
<td>basic</td>
<td>5</td>
<td><span color="green">Done</span></td>
<td>`results/passat_arxiv/basic_qwen3_4b_topk5.jsonl`</td>
</tr>
<tr>
<td>basic</td>
<td>10</td>
<td><span color="green">Done</span></td>
<td>`results/passat_arxiv/basic_qwen3_4b_topk10.jsonl`</td>
</tr>
<tr>
<td><span color="green">one\_search</span></td>
<td>5</td>
<td><span color="green">Done</span></td>
<td>`results/passat_arxiv/one_search_qwen3_4b_topk5.jsonl`</td>
</tr>
<tr>
<td><span color="green">one\_search</span></td>
<td>10</td>
<td><span color="green">Done</span></td>
<td>`results/passat_arxiv/one_search_qwen3_4b_topk10.jsonl`</td>
</tr>
<tr>
<td>reviewer\_turns</td>
<td>5</td>
<td><span color="green">Done</span></td>
<td>`results/passat_arxiv/reviewer_turns_qwen3_4b_topk5.jsonl`</td>
</tr>
<tr>
<td>reviewer\_turns</td>
<td>10</td>
<td><span color="green">Done</span></td>
<td>`results/passat_arxiv/reviewer_turns_qwen3_4b_topk10.jsonl`</td>
</tr>
<tr>
<td>claim\_contrast</td>
<td>5</td>
<td><span color="green">Done</span></td>
<td>`results/passat_arxiv/claim_contrast_qwen3_4b_topk5.jsonl`</td>
</tr>
<tr>
<td>claim\_contrast</td>
<td>10</td>
<td><span color="green">Done</span></td>
<td>`results/passat_arxiv/claim_contrast_qwen3_4b_topk10.jsonl`</td>
</tr>
<tr>
<td>prior\_art</td>
<td>5</td>
<td><span color="green">Done</span></td>
<td>`results/passat_arxiv/prior_art_qwen3_4b_topk5.jsonl`</td>
</tr>
<tr>
<td>prior\_art</td>
<td>10</td>
<td><span color="green">Done</span></td>
<td>`results/passat_arxiv/prior_art_qwen3_4b_topk10.jsonl`</td>
</tr>
<tr>
<td>debate\_search</td>
<td>5</td>
<td><span color="green">Done</span></td>
<td>`results/passat_arxiv/debate_search_qwen3_4b_topk5.jsonl`</td>
</tr>
<tr>
<td>debate\_search</td>
<td>10</td>
<td><span color="green">Done</span></td>
<td>`results/passat_arxiv/debate_search_qwen3_4b_topk10.jsonl`</td>
</tr>
<tr>
<td>structured\_evidence</td>
<td>5</td>
<td><span color="green">Done</span></td>
<td>`results/passat_arxiv/structured_evidence_qwen3_4b_topk5.jsonl`</td>
</tr>
<tr>
<td>structured\_evidence</td>
<td>10</td>
<td><span color="green">Done</span></td>
<td>`results/passat_arxiv/structured_evidence_qwen3_4b_topk10.jsonl`</td>
</tr>
<tr>
<td><span color="green">one\_search (full test)</span></td>
<td>10</td>
<td><span color="green">Done</span></td>
<td>`results/passat_arxiv/one_search_qwen3_4b_topk10_full.jsonl` (466 MB)</td>
</tr>
</table>

## Figures

<image source="https://raw.githubusercontent.com/SachinKonan/SkyRL/searchenvopt/skyrl-train/results/passat_arxiv/prompt_ablation_arxiv.png">ArXiv Prompt Ablation — pass@k, majority@k, threshreject, threshaccept across 7 prompts x 2 topk values</image>

## Key Files

- **Sbatch (ablation)**: `examples/search/sbatch/run_base_eval_prompt_ablation_arxiv.sbatch`
- **Sbatch (full test)**: `examples/search/sbatch/run_one_search_full.sbatch`
- **Inference script**: `scripts/vllm_infer.sh`
- **Plotting script**: `scripts/plot_pass_at_k.py`
- **Results dir**: `results/passat_arxiv/`
- **Full test results**: `results/passat_arxiv/one_search_qwen3_4b_topk10_full.jsonl` (466 MB)
- **Config**: array job 0-13 (7 prompts x 2 topk), 2x GPU (PLI), 30 completions/paper, max\_gen\_len=800

## Conclusion

- **`one_search` is the best ArXiv prompt** — achieves highest threshaccept on the small test set (~71%) with a simple 2-turn structure (search once, then decide).
- topk=10 generally outperforms topk=5, providing more retrieval context.
- Multi-turn prompts (4 turns) don't improve over the simpler 2-turn `one_search` — additional search rounds add noise without improving decision quality.
- **Scaling caveat**: The `one_search` prompt performs well on the small test set but **does not scale** to the full 1500-paper test set — performance degrades, suggesting overfitting to the test set distribution or retrieval quality issues at scale.
- `one_search` with topk=10 was selected as the base for ArXiv RL training (Task 5).

▶ Sbatch Configuration
	**Job name**: `arxiv_prompt_ablation`
	**Array**: 0-13 (prompt\_idx \* 2 + topk\_idx)
	**GPUs**: 2x GPU (PLI partition)
	**Memory**: 80G, 6 CPUs
	**Time**: 8:00:00
	**Model**: Qwen3-4B-Thinking only
	**Prompts**: basic, one\_search, reviewer\_turns, claim\_contrast, prior\_art, debate\_search, structured\_evidence
	**TopK**: 5, 10
	**Max turns**: 2 for one\_search, 4 for all others
	**N samples**: 30 completions/paper
	**Max gen length**: 800
	**Eval batch size**: 16
