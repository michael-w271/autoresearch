# autoresearch — Repository Summary

**Source**: [https://github.com/karpathy/autoresearch](https://github.com/karpathy/autoresearch)  
**Fork**: [https://github.com/michael-w271/autoresearch](https://github.com/michael-w271/autoresearch)  
**Author**: Andrej Karpathy, March 2026  
**License**: MIT

---

## What It Is

A minimal, self-contained framework for **autonomous LLM pre-training research**.
You give an AI agent one Python file (`train.py`), a fixed 5-minute time budget per
experiment, and a set of written instructions (`program.md`). The agent modifies the
training code, runs it, evaluates the result, and iterates — continuously, overnight.
You wake up to a log of ~100 experiments and (hopefully) a better model.

> *"You are not touching any of the Python files like you normally would as a
> researcher. Instead, you are programming the `program.md` Markdown files..."*
> — README

The only metric is **val_bpb** (validation bits per byte). Lower is better. Because
the time budget is fixed, changes in model size, batch size, or architecture are
directly comparable — the agent is searching for the best model *for your specific
hardware* in 5 minutes.

---

## Repo Layout

```
autoresearch/
├── prepare.py     — fixed constants, one-time data download, BPE tokenizer,
│                    dataloader, evaluation harness. NEVER modified by the agent.
├── train.py       — GPT model + MuonAdamW optimizer + training loop.
│                    THE ONLY file the agent edits.
├── program.md     — agent instructions ("the research org code"). Edited by the human.
├── analysis.ipynb — Jupyter notebook for plotting/analysing results.tsv
├── progress.png   — benchmark progress chart (teaser image in README)
└── pyproject.toml — uv-managed dependencies (PyTorch 2.9.1 + CUDA 12.8)
```

---

## Key Files

### [`prepare.py`](prepare.py) — Fixed harness (do not modify)

Sets the immutable experiment constants:

| Constant | Value | Meaning |
|----------|-------|---------|
| `MAX_SEQ_LEN` | 2048 | Transformer context length |
| `TIME_BUDGET` | 300 s | 5-minute wall-clock training budget |
| `EVAL_TOKENS` | ~20M | Tokens used for validation loss |
| `VOCAB_SIZE` | 8192 | BPE vocabulary size |

**Dataset**: [`karpathy/climbmix-400b-shuffle`](https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle) — streamed as Parquet shards from HuggingFace.  
**Tokenizer**: Custom BPE using `rustbpe`, GPT-4-style split pattern, trained on the first data shard, stored under `~/.cache/autoresearch/`.  
**Validation**: shard `06542` is pinned as the fixed validation set; `evaluate_bpb()` reports bits-per-byte — vocabulary-size-independent and architecture-agnostic, so all experiments are fairly compared.

---

### [`train.py`](train.py) — The agent's sandbox

A ~630-line single-file GPT implementation, simplified from Karpathy's
[nanochat](https://github.com/karpathy/nanochat). The baseline model:

#### Architecture

| Parameter | Default | Notes |
|-----------|---------|-------|
| `n_layer` | 12 | Transformer depth |
| `n_embd` | 768 | Hidden dimension |
| `n_head` | 6 | Query heads |
| `n_kv_head` | 6 | Key/Value heads (supports GQA if reduced) |
| `vocab_size` | 32768 | Embedding vocab size |
| `sequence_len` | 2048 | Context length |
| `window_pattern` | `"SSSL"` | Alternating banded + full attention |

**Notable architecture features** (not in vanilla GPT-2):

- **RMSNorm** (`F.rms_norm`) everywhere — no LayerNorm bias
- **RoPE** (Rotary Position Embeddings) with precomputed cos/sin
- **Flash Attention 3** via the `kernels` package — automatically picks the
  Hopper FA3 kernel (`varunneal/flash-attention-3`) or the community fork
  (`kernels-community/flash-attn3`) based on `torch.cuda.get_device_capability()`
- **Sliding window attention** — `window_pattern = "SSSL"` means 3 of every 4
  layers use a `T/2`-token banded window; the last layer always uses full context
- **Value Embeddings** (ResFormer-style): every other layer adds a learned per-token
  value embedding with an input-dependent gate, letting the model build
  token-specific "memory" into attention values
- **Learnable per-layer residual scalars** `resid_lambdas` and `x0_lambdas` that
  mix the previous layer's output with a scaled copy of `x0` (the embedding)
- **Logit softcapping**: `15 * tanh(logits / 15)` — prevents logit explosion
- **Relu²** activation in MLP (not GELU)

#### Optimizer: MuonAdamW

A hybrid optimizer that applies different update rules per parameter group:

| Parameter group | Optimizer | Default LR |
|----------------|-----------|------------|
| Transformer weight matrices | **Muon** (orthogonal gradient) | 0.02 |
| Embeddings (wte, value embeds) | AdamW | 0.2 |
| Unembedding (lm_head) | AdamW | 0.004 |
| Residual scalars (`resid_lambdas`) | AdamW | 0.005 |
| x0 lambdas | AdamW | 0.5 |

**Muon** uses the *Polar Express* orthogonalization (5-step Newton iteration) to
project gradients onto the orthogonal group before applying Nesterov momentum. It
also includes **NorMuon** variance reduction and cautious weight decay. Both the AdamW
and Muon step functions are `@torch.compile`d for performance.

LR scaling follows $1/\sqrt{d_{model}/768}$ for AdamW groups (Maximal Update
Parameterization style).

---

### [`program.md`](program.md) — The agent's instructions

The human-editable "research org specification". It tells the AI agent exactly how
to run experiments:

1. **Setup**: create a branch `autoresearch/<tag>`, read all source files, verify
   data cache, initialise `results.tsv`.
2. **Loop forever**:
   - Propose one change to `train.py`
   - `git commit` the change
   - Run `uv run train.py > run.log 2>&1`
   - Parse `val_bpb` and `peak_vram_mb` from the log
   - Log to `results.tsv` with status `keep` / `discard` / `crash`
   - Decide whether to keep the change based on val_bpb delta and code complexity
3. **Constraints**: only `train.py` may be modified; no new packages; 5-minute budget
   is sacrosanct; simplicity is a first-class objective.

`results.tsv` schema:
```
commit  val_bpb  memory_gb  status  description
```

---

## Workflow

```bash
# Dependencies (uv project manager required)
uv sync

# One-time data prep (~2 min, downloads data shards + trains tokenizer)
uv run prepare.py

# Manual single run (5 min)
uv run train.py

# Autonomous agent mode — point Claude/Codex at the repo, then:
# "Hi have a look at program.md and let's kick off a new experiment!"
```

---

## Design Philosophy

| Principle | Implementation |
|-----------|---------------|
| **One file, one metric** | Agent only touches `train.py`; only `val_bpb` matters |
| **Fixed time budget** | 5 min wall clock → ~12 experiments/hour, ~100 overnight |
| **Platform-relative results** | Experiments are comparable *on your hardware*, not across machines |
| **Simplicity as a reward** | `program.md` explicitly rewards removing code ≈ keeping code that achieves equal loss |
| **Self-contained** | Single GPU, no distributed training, no configs, no Hydra/wandb |

---

## Dependencies

Managed by [uv](https://docs.astral.sh/uv/), pinned in [pyproject.toml](pyproject.toml):

| Package | Version | Purpose |
|---------|---------|---------|
| `torch` | 2.9.1+cu128 | Training |
| `kernels` | ≥0.11.7 | Flash Attention 3 dispatch |
| `rustbpe` | ≥0.1.0 | Fast BPE tokenizer |
| `tiktoken` | ≥0.11.0 | GPT-4 split pattern |
| `pyarrow` | ≥21.0.0 | Parquet shard loading |
| `matplotlib` / `pandas` | — | Analysis notebook |

---

## GPU Notes

- **Officially tested**: NVIDIA H100 (sm_90, Hopper). Flash Attention 3 Hopper path
  uses `varunneal/flash-attention-3`.
- **Other NVIDIA GPUs** (Ada, Ampere, Blackwell): community FA3 path
  `kernels-community/flash-attn3` is used automatically.
- **RTX 5080 (sm_120)**: should work via the community FA3 fallback, but is not
  tested in the upstream repo. Worth verifying before running overnight.
- **Non-NVIDIA (CPU, MPS)**: not supported in this repo. See the
  [notable forks](#notable-forks) below.

---

## Notable Forks

| Fork | Platform |
|------|----------|
| [miolini/autoresearch-macos](https://github.com/miolini/autoresearch-macos) | macOS |
| [trevin-creator/autoresearch-mlx](https://github.com/trevin-creator/autoresearch-mlx) | macOS / MLX |
| [jsegov/autoresearch-win-rtx](https://github.com/jsegov/autoresearch-win-rtx) | Windows + RTX |

---

## Relevance to This Project

autoresearch is a close philosophical parallel to our **CUDA agent loop**
(`scripts/cuda_agent_loop.py`):

| autoresearch | CUDA-sparse agent loop |
|-------------|----------------------|
| Agent edits `train.py` | Agent edits `scripts/kernels/sparse_14_gemm.cu` |
| Metric: `val_bpb` (lower) | Metric: TFLOP/s (higher) |
| Fixed 5-min time budget per run | Fixed compile + bench per iteration |
| `program.md` as skill file | `CUDA-Agent-BytedTsinghua/agent_workdir/SKILL.md` |
| `results.tsv` log | `results/sparse_14_kernel/agent_loop/summary.json` |
| Human iterates on `program.md` | Human iterates on SKILL.md + loop params |

The key difference: autoresearch targets **training quality** (model expressiveness);
our loop targets **hardware throughput** (GEMM performance). Both use the same
agentic "compile → measure → improve → repeat" pattern described in CUDA-Agent
(arxiv 2602.24286).

The Muon optimizer and value embeddings (ResFormer) used in `train.py` are also
relevant reading for our S-STE / CHTs24 training work — particularly the per-layer
learnable residual scalars and the logit softcapping technique.
