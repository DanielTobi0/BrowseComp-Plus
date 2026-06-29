# Running BrowseComp-Plus with Qwen3.5-9B on a Single RTX 4090

This documents everything needed to run the BrowseComp-Plus benchmark using
`Qwen/Qwen3.5-9B` as the search agent on a single RTX 4090 (24 GB VRAM).

---

## One-time fixes applied to this repo

These changes were made once and are already committed to the working tree.

### 1. Install Java 21 (required by pyserini/BM25)

```bash
apt update && apt install -y openjdk-21-jdk
```

### 2. Upgrade vllm to support Qwen3.5

`Qwen/Qwen3.5-9B` uses the `Qwen3_5ForConditionalGeneration` architecture, which
requires vllm ≥ 0.23.0. The original `pyproject.toml` pinned `vllm==0.9.0.1`.

```bash
source .venv/bin/activate
uv pip install "vllm==0.23.0"
```

This also upgraded pydantic to 2.13.4, which broke `fastmcp==2.9.2`.

### 3. Upgrade fastmcp to work with pydantic 2.13.x

```bash
uv pip install "fastmcp==2.14.7"
```

### 4. Patch pyserini's OpenAI import

`pyserini` initialises an `openai.OpenAI()` client at import time with an empty
string API key, which the newer openai SDK rejects. Fixed in:

```
.venv/lib/python3.10/site-packages/pyserini/encode/_openai.py  line ~27
```

Changed `api_key = ''` → `api_key = os.getenv("OPENAI_API_KEY") or "dummy"`.

### 5. Make FAISS/tevatron imports lazy in the searcher package

`vllm==0.23.0` pulled in `transformers==5.x`, which is incompatible with the
pinned `peft` version. The `searchers/__init__.py` eagerly imported `FaissSearcher`
(which depends on tevatron → peft), causing every `mcp_server.py` startup to fail
even when only BM25 was requested.

Fixed by making the FAISS and ReasonIR imports lazy (loaded on demand only when
those searcher types are actually selected). See `searcher/searchers/__init__.py`.

---

## Running the benchmark

Run the three commands below **in separate terminals** (or with `nohup … &` to
background them). Start them in order and wait for each to be ready before
starting the next.

### Step 1 — Serve the LLM

```bash
source .venv/bin/activate

vllm serve Qwen/Qwen3.5-9B \
    --port 8000 \
    --max-model-len 32768 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.92 \
    --quantization fp8 \
    --trust-remote-code
```

Wait until the log prints `Application startup complete` (~4–5 min on first run
due to torch.compile; faster on subsequent runs once the cache is warm).

**Why FP8?** The model weights alone use ~21 GB in bf16, leaving no room for KV
cache on a 24 GB card. FP8 halves the weight footprint to ~11 GB, freeing ~13 GB
for the KV cache and allowing a 32 K context window.

### Step 2 — Start the BM25 retrieval server

```bash
source .venv/bin/activate

python searcher/mcp_server.py \
    --searcher-type bm25 \
    --index-path indexes/bm25 \
    --port 8080 \
    --transport sse
```

This uses the defaults: `--snippet-max-tokens 512` and `--k 5` (5 documents per
search, each truncated to 512 tokens). Each search round adds ~2,560 tokens to
the context.

> **Context budget note:** with a 32 K window, 10 K output tokens, and a ~6,400
> token initial prompt, there is room for roughly **6 search rounds** at default
> settings before the context fills up. Queries that require more rounds will
> fail with a 400 error and be skipped. If you need more rounds (the paper's
> best models average 20+), reduce snippet size: add
> `--snippet-max-tokens 256 --k 3` to fit ~21 rounds.

### Step 3 — Run the agent over all 830 queries

```bash
source .venv/bin/activate

python search_agent/qwen_client.py \
    --model Qwen/Qwen3.5-9B \
    --model-server http://localhost:8000/v1 \
    --mcp-url "http://127.0.0.1:8080/mcp" \
    --output-dir runs/bm25/qwen3.5-9b \
    --query topics-qrels/queries.tsv \
    --query-template QUERY_TEMPLATE_NO_GET_DOCUMENT
```

Results are saved incrementally to `runs/bm25/qwen3.5-9b/run_*.json`.
If the run is interrupted, restarting the same command will skip already-processed
query IDs automatically.

**Expected rate:** ~50–70 s/query on a single RTX 4090 → ~14 hours for all 830.

> **Note on `--max_tokens`:** The default is 10,000. Some queries that require
> many search rounds will hit a 400 context-overflow error and be skipped; the
> run continues and the checkpoint logic means they can be retried. This is
> expected behaviour at default settings on a 32 K context window.

---

## Evaluation

Once the run is complete, evaluate with a judge model. The paper uses **gpt-4.1**
(OpenAI API); the script defaults to **Qwen/Qwen3-32B** as a local fallback.

### Option A — gpt-4.1 (matches leaderboard scores)

Requires an OpenAI API key and a small code change to `evaluate_run.py` to call
the API instead of local vllm.

### Option B — Qwen/Qwen3-8B (fits on the RTX 4090, free)

First stop the vllm server from Step 1, then:

```bash
source .venv/bin/activate

python scripts_evaluation/evaluate_run.py \
    --input_dir runs/bm25/qwen3.5-9b \
    --model Qwen/Qwen3-8B \
    --tensor_parallel_size 1
```

Use the **same judge model** for both the baseline and the fine-tuned run so the
relative improvement is comparable.

---

## Leaderboard submission

After evaluation, edit the generated `evals/.../evaluation_summary.json` and fill
in the `"LLM"`, `"Retriever"`, and `"Link"` fields before submitting.
