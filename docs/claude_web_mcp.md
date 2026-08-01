# Using the retrieval index from Claude web (custom connector)

This exposes the repo's hybrid retriever (BM25 + dense, RRF-fused, cross-encoder
reranked) as a remote MCP server, so you can paste a BrowseComp-Plus question into
a Claude web chat and have Claude search the local corpus itself — the same tools
(`search`, `get_document`) the local agents get, driven by a much stronger model.

The point is diagnosis: when a question fails locally, you want to know *whether
the evidence was retrievable at all*. If Claude web, with the same index, finds the
answer, the failure was the agent's search strategy or reasoning. If Claude can't
find it either, the retriever is the bottleneck.

## 1. Start the server

```bash
scripts_mcp/register_connector.sh
```

This installs a supervisor service (`bcp-mcp`) that runs
[`scripts_mcp/serve_claude_web.sh`](../scripts_mcp/serve_claude_web.sh), waits for
the models to load, opens a Cloudflare quick tunnel, and prints the connector URL:

```
https://<random>.trycloudflare.com/<secret>/mcp
```

Because the service is under supervisor it survives your shell closing and restarts
on crash. Logs: `tail -f /var/log/portal/bcp-mcp.log`.

Defaults (override with env vars, see the top of `serve_claude_web.sh`):

| Var | Default | Notes |
| --- | --- | --- |
| `MCP_PORT` | `10100` | Free open port on this instance; bound to `127.0.0.1` only |
| `MCP_K` | `5` | Results per search — same k the eval runs used |
| `MCP_SNIPPET_MAX_TOKENS` | `512` | Snippet truncation; `-1` for full text |
| `MCP_EMBEDDING_MODEL` | `Qwen/Qwen3-Embedding-0.6B` | Set to `Qwen/Qwen3-Embedding-8B` to match the paper |
| `MCP_FAISS_INDEX` | `indexes/qwen3-embedding-0.6b/corpus.shard*_of_4.pkl` | Match the model above |
| `MCP_BM25_INDEX` | `indexes/bm25` | |

To reproduce a specific failed run faithfully, use the same `MCP_K`,
`MCP_SNIPPET_MAX_TOKENS` and index the run used — retrieval settings change the
answer as much as the model does.

### Security model — read this before pasting the URL anywhere

Claude web custom connectors cannot send an `Authorization` header, so **the random
path segment in the URL is the credential**. Anyone with the full URL can query your
index until the tunnel dies. That is acceptable for a public benchmark corpus; do
not reuse this pattern for private documents without real auth (FastMCP supports
OAuth, or put the server behind the instance's Caddy auth edge and use a client that
can send tokens).

The server binds `127.0.0.1`, so the tunnel is the only way in. `supervisorctl stop
bcp-mcp` closes it. Quick tunnels are ephemeral — after a reboot, re-run
`register_connector.sh` to get the new URL and update the connector in Claude.

## 2. Add it in Claude web

1. Settings → Connectors → **Add custom connector**.
2. Name: `BrowseComp-Plus retrieval`. URL: the printed
   `https://….trycloudflare.com/<secret>/mcp`.
3. No OAuth — leave the client ID/secret blank; the server does not challenge.
4. Open a new chat, enable the connector, and confirm `search` and `get_document`
   appear in the tool list.

Sanity check before real questions: *"Use the search tool for `Queen Arwa University
cultural week` and show me the docids you get back."* You should get five hits with
docids and snippets.

## 3. The prompt — use the benchmark's, verbatim

To keep Claude web's answers comparable to your local runs, paste the benchmark's
own prompt with the question substituted in. It is `QUERY_TEMPLATE` from
[`search_agent/prompts.py`](../search_agent/prompts.py) — the get_document variant,
which matches the two tools this connector exposes.

Copy this whole block into the chat, replacing the `Question:` line:

```
You are a deep research agent. You need to answer the given question by interacting with a search engine, using the search and get_document tools provided. Please perform reasoning and use the tools step by step, in an interleaved manner. You may use the search and get_document tools multiple times.

Question: <paste the failed question here>

Your response should be in the following format:
Explanation: {your explanation for your final answer. For this explanation section only, you should cite your evidence documents inline by enclosing their docids in square brackets [] at the end of sentences. For example, [20].}
Exact Answer: {your succinct, final answer}
Confidence: {your confidence score between 0% and 100% for your answer}
```

Three details that keep this faithful to the harness:

- **The braces stay single.** In `prompts.py` they are doubled (`{{...}}`) only to
  survive `str.format`; the string the model actually receives has single braces,
  exactly as above.
- **No system prompt.** The clients send this template as the entire user message
  and set no system prompt unless `--system` is passed, which the runs did not
  ([`search_agent/qwen_client.py:50-61`](../search_agent/qwen_client.py#L50-L61)).
  So leave Claude's project instructions empty — anything you put there is an extra
  variable the local runs didn't have.
- **Use a fresh chat per question.** Each benchmark query is an independent
  conversation; reusing a chat leaks documents and reasoning from the previous
  question into the next one.

If you want the search-only variant (no `get_document`), that is
`QUERY_TEMPLATE_NO_GET_DOCUMENT` in the same file — but then start the server
without `--get-document` so the tools match the prompt.

### Grading the answer the same way

Because the output format is identical, a Claude web response can go straight
through the benchmark's judge. `GRADER_TEMPLATE` in `prompts.py` is what
`scripts_evaluation/evaluate_with_openai.py` uses — the judge sees the question,
the response, and the gold `correct_answer`, and returns
`extracted_final_answer` / `reasoning` / `correct` / `confidence`. Paste the
response in there rather than eyeballing the match, so "Claude got it" means the
same thing it meant in your 57.3% number.

### What to watch for

This is where a stronger model behaves differently from the benchmarked one, and
it affects how you read the result:

- **Answering without searching.** Claude recognises many BrowseComp-Plus entities
  from pretraining and can answer with zero tool calls. That result is worthless
  for diagnosing retrieval — check the tool calls actually happened, and redo the
  question if they didn't. The benchmark prompt has no "only use the tools" clause
  to prevent this, so it is on you to notice.
- **Which failure you're diagnosing.** If Claude finds the answer with the same
  index, retrieval was sufficient and the local agent's search strategy or
  reasoning was the gap. If Claude can't find it either, the retriever is the
  bottleneck. Comparing the docids Claude cites against `retrieved_docids` in the
  run file and the gold qrels in `topics-qrels/` tells you which.
- **Deviating on purpose.** Once you've recorded the verbatim-prompt result, extra
  instructions ("try more query phrasings", "call get_document before relying on a
  snippet") are a useful second experiment — just don't mix the two, since they no
  longer measure the same thing as the benchmark.

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| Claude can't connect | Tunnel died — re-run `register_connector.sh`, update the URL in Settings → Connectors |
| Connector added, no tools | Wait for model load (`tail -f /var/log/portal/bcp-mcp.log`), then reconnect in Claude |
| Every search is slow | Reranker runs on GPU; check nothing else is holding VRAM (`nvidia-smi`) |
| Answers cite docids not in the corpus | Claude is answering from memory — restate the "only use the tools" rule |
| Want the paper's exact retriever | Set `MCP_EMBEDDING_MODEL=Qwen/Qwen3-Embedding-8B` and `MCP_FAISS_INDEX='indexes/qwen3-embedding-8b/corpus.shard*_of_4.pkl'`, then `supervisorctl restart bcp-mcp` (first start downloads ~16 GB) |
