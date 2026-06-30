# Running BrowseComp-Plus with Qwen3.5-9B on an A100 80GB

This documents everything needed to run the BrowseComp-Plus benchmark using
`Qwen/Qwen3.5-9B` as the search agent on an A100 80GB PCIe, evaluate with a
Together AI judge (Gemma model), and compare a baseline against a fine-tuned model.

All commands assume you are in the repo root:
```bash
cd /workspace/BrowseComp-Plus
source .venv/bin/activate
```

---

## One-time fixes applied to this repo

These changes were made once and are already in the working tree — no need to
repeat them.

### 1. Java 21 (required by pyserini / BM25)

```bash
apt update && apt install -y openjdk-21-jdk
```

### 2. Upgrade vllm to support Qwen3.5

`Qwen/Qwen3.5-9B` uses the `Qwen3_5ForConditionalGeneration` architecture,
which requires vllm ≥ 0.23.0 (original pin was `vllm==0.9.0.1`).

```bash
uv pip install "vllm==0.23.0"
```

This pulled in pydantic 2.13.4, which broke `fastmcp==2.9.2`.

### 3. Upgrade fastmcp

```bash
uv pip install "fastmcp==2.14.7"
```

### 4. Patch pyserini's OpenAI import

`pyserini` initialises an `openai.OpenAI()` client at import time using an empty
string API key, which the newer openai SDK rejects. Patched in:

```
.venv/lib/python3.10/site-packages/pyserini/encode/_openai.py  (~line 27)
```

`api_key = ''` → `api_key = os.getenv("OPENAI_API_KEY") or "dummy"`

### 5. Lazy FAISS imports in the searcher package

vllm 0.23.0 pulled in `transformers==5.x`, incompatible with the pinned `peft`.
`searcher/searchers/__init__.py` eagerly imported `FaissSearcher` (which chains
to tevatron → peft), crashing every `mcp_server.py` startup even when only BM25
was needed. Fixed by making FAISS and ReasonIR imports lazy. See
`searcher/searchers/__init__.py`.

---

## Benchmark run

Start the three processes **in order**, each in its own terminal (or backgrounded
with `nohup … &`). Wait for each to be ready before starting the next.

### Step 1 — Serve the LLM

```bash
vllm serve Qwen/Qwen3.5-9B \
    --port 8000 \
    --max-model-len 260000 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.92 \
    --trust-remote-code
```

Ready when the log prints `Application startup complete` (~4–5 min on first run
due to torch.compile; faster once the compile cache is warm).

**Why no quantization?** The A100 has 80 GB VRAM. Qwen3.5-9B in BF16 uses ~18 GB
for weights, leaving ~55 GB for the KV cache — more than enough for the full 128 K
(`131072`) context window at full precision.

### Step 2 — Start the BM25 retrieval server

```bash
python searcher/mcp_server.py \
    --searcher-type bm25 \
    --index-path indexes/bm25 \
    --port 8080 \
    --transport sse
```

Uses defaults: `--snippet-max-tokens 512`, `--k 5` (5 documents per search,
each up to 512 tokens). Each search round adds ~2,560 tokens to the context.

> **Context budget:** with a 128 K window, ~10 K output tokens, and a ~6,400
> token initial prompt, there is room for roughly **44 search rounds** at default
> settings — far more than earlier 32 K setups. Context exhaustion is no longer
> a practical concern at these settings.

### Step 3 — Run the agent over all 830 queries

```bash
python search_agent/qwen_client.py \
    --model Qwen/Qwen3.5-9B \
    --model-server http://localhost:8000/v1 \
    --mcp-url "http://127.0.0.1:8080/mcp" \
    --output-dir runs/bm25/qwen3.5-9b \
    --query topics-qrels/queries.tsv \
    --query-template QUERY_TEMPLATE_NO_GET_DOCUMENT
```

Results are written incrementally to `runs/bm25/qwen3.5-9b/run_*.json`.
Restarting the same command automatically skips already-saved query IDs.

**Expected rate:** ~15–25 s/query → ~4–6 hours for all 830 on an A100 80GB.

---

## Evaluation — Together AI (Gemma)

All 830 judgements run in parallel via the Together AI API. No local GPU needed.

```bash
export TOGETHER_API_KEY="your_key_here"

python scripts_evaluation/evaluate_run.py \
    --input_dir runs/bm25/qwen3.5-9b \
    --model google/gemma-4-31B-it \
    --num-workers 8
```

`TOGETHER_API_KEY` is read from the environment automatically — no need to pass
`--together-api-key` explicitly if the env var is set.  
`--num-workers` controls parallel API threads (default 8).

> Use the **same judge model** for both the baseline and the fine-tuned run so
> the relative improvement is a fair comparison.

---

## Full workflow: baseline → fine-tune → compare

```
1. Run benchmark (Step 1–3 above)          → runs/bm25/qwen3.5-9b/
2. Evaluate baseline                        → evals/bm25/qwen3.5-9b/evaluation_summary.json
3. Fine-tune Qwen3.5-9B on your dataset
4. Re-run benchmark with fine-tuned model   → runs/bm25/qwen3.5-9b-finetuned/
5. Evaluate fine-tuned model (same judge)   → evals/bm25/qwen3.5-9b-finetuned/evaluation_summary.json
6. Compare Accuracy (%) between the two summaries
```

For Step 4, swap `--model` and `--output-dir` in the qwen_client.py command to
point at the fine-tuned checkpoint and a new output directory.

---

## Leaderboard submission

After evaluation, fill in the placeholder fields in
`evals/.../evaluation_summary.json` before submitting:

```json
{
  "LLM": "Qwen/Qwen3.5-9B",
  "Retriever": "BM25",
  "Link": "https://huggingface.co/..."
}
```
