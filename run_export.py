"""Look up run records by query id and render them as markdown or print-ready PDFs.

Each run under ``runs/<retriever>/<model>/`` is a single JSON file named after its
timestamp, so the ``query_id`` it belongs to is only visible inside the file — hence
the lookup helpers that scan a run directory and hand back the matching record.

A record holds a flat ``result`` list of reasoning blocks, tool calls and the final
answer, with every tool payload stashed as an escaped JSON string. That is unreadable
as raw JSON, so the rendering helpers turn a record into a markdown transcript: query
text and gold docids up front, then the trace step by step, with long tool outputs
folded into ``<details>`` blocks.

Those folds are right on screen and useless on paper — a collapsed block simply does
not print. The PDF path therefore takes the same markdown, unfolds every block into a
labelled section, applies a print stylesheet, and hands the result to WeasyPrint.

    python run_export.py 769                      # one query -> stdout
    python run_export.py 769 1049 -o trace.md     # several -> one file
    python run_export.py --all -o markdown_runs/  # every run -> one markdown file each
    python run_export.py --pdf 6 778              # one PDF per query
    python run_export.py --samples 2              # 2 passed + 2 failed, plus a combined PDF
    python run_export.py --outline 769            # just the step types, to see what a run did

From a notebook, pass just the query id:

    save_markdown(250)   # writes markdown_runs/query_250.md, returns the path
    show_markdown(250)   # renders the transcript inline

The PDF half needs ``weasyprint`` and ``markdown`` (``uv pip install weasyprint
markdown``); the markdown half needs neither, so they are imported only on use.
"""

import json
import re
from pathlib import Path

DEFAULT_RUNS_DIR = Path(__file__).parent / "runs" / "custom" / "qwen3.5-9b"

QUERIES_FILE = Path(__file__).parent / "topics-qrels" / "queries.tsv"
GOLDS_FILE = Path(__file__).parent / "topics-qrels" / "qrel_golds.txt"
EVIDENCE_FILE = Path(__file__).parent / "topics-qrels" / "qrel_evidence.txt"
EVAL_CSV = Path(__file__).parent / "evals" / "custom" / "qwen3.5-9b" / "detailed_judge_results.csv"

DEFAULT_OUT_DIR = Path(__file__).parent / "markdown_runs"
DEFAULT_PDF_DIR = DEFAULT_OUT_DIR / "pdf"

SNIPPET_CHARS = 600
DOCUMENT_CHARS = 3000

# Tighter than the on-screen defaults: every character costs a page on paper.
PDF_SNIPPET_CHARS = 400
PDF_DOCUMENT_CHARS = 1500


# --- Finding a run by query id ----------------------------------------------


def iter_runs(runs_dir=DEFAULT_RUNS_DIR):
    """Yield ``(path, record)`` for every run JSON in ``runs_dir``."""
    for path in sorted(Path(runs_dir).glob("*.json")):
        with open(path) as f:
            yield path, json.load(f)


def build_index(runs_dir=DEFAULT_RUNS_DIR):
    """Map ``query_id`` -> list of file paths, newest last (files sort by timestamp).

    A query can appear more than once if a run was repeated, hence the list.
    """
    index = {}
    for path, record in iter_runs(runs_dir):
        query_id = record.get("query_id")
        if query_id is not None:
            index.setdefault(str(query_id), []).append(path)
    return index


def find_run(query_id, runs_dir=DEFAULT_RUNS_DIR):
    """Return the full run record for ``query_id``, or ``None`` if absent.

    If several runs share the query, the most recent file wins.
    """
    match = None
    for path, record in iter_runs(runs_dir):
        if str(record.get("query_id")) == str(query_id):
            match = record
    return match


def get_result(query_id, runs_dir=DEFAULT_RUNS_DIR, default=None):
    """Return the ``result`` list for ``query_id`` (``default`` if not found)."""
    record = find_run(query_id, runs_dir)
    if record is None:
        return default
    return record.get("result", default)


def get_results(query_ids, runs_dir=DEFAULT_RUNS_DIR):
    """Return ``{query_id: result}`` for many ids in a single pass over the files."""
    wanted = {str(q) for q in query_ids}
    found = {}
    for _, record in iter_runs(runs_dir):
        query_id = str(record.get("query_id"))
        if query_id in wanted:
            found[query_id] = record.get("result")
    return found


# --- Topics and qrels -------------------------------------------------------


def load_queries(path=QUERIES_FILE):
    """Map ``query_id`` -> query text from the tab-separated topics file."""
    queries = {}
    if not Path(path).exists():
        return queries
    with open(path) as f:
        for line in f:
            query_id, _, text = line.partition("\t")
            if text:
                queries[query_id.strip()] = text.strip()
    return queries


def load_qrels(path):
    """Map ``query_id`` -> set of judged docids from a TREC-style qrels file."""
    qrels = {}
    if not Path(path).exists():
        return qrels
    with open(path) as f:
        for line in f:
            fields = line.split()
            if len(fields) >= 4 and fields[3] != "0":
                qrels.setdefault(fields[0], set()).add(fields[2])
    return qrels


def shared_tables():
    """Load queries and qrels once so batch exports don't re-read them per run.

    Pass the result straight into ``record_to_markdown``/``run_to_markdown`` as kwargs.
    """
    return {"queries": load_queries(), "golds": load_qrels(GOLDS_FILE), "evidence": load_qrels(EVIDENCE_FILE)}


# --- Markdown transcripts ---------------------------------------------------


def _truncate(text, limit):
    """Clip ``text`` to ``limit`` characters, noting how much was dropped."""
    text = (text or "").strip()
    if limit is None or len(text) <= limit:
        return text
    return text[:limit].rstrip() + f"\n\n… [{len(text) - limit} more characters]"


def _details(summary, body, open_by_default=False):
    """Wrap ``body`` in a collapsible block (blank lines keep markdown rendering)."""
    flag = " open" if open_by_default else ""
    return f"<details{flag}>\n<summary>{summary}</summary>\n\n{body}\n\n</details>"


def _fence(text, lang=""):
    """Fence ``text`` as a code block, widening the fence if it contains backticks."""
    fence = "```"
    while fence in text:
        fence += "`"
    return f"{fence}{lang}\n{text}\n{fence}"


def _format_arguments(raw):
    """Pretty-print a tool call's JSON argument string, or pass it through as-is."""
    try:
        return json.dumps(json.loads(raw), indent=2, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(raw)


def _format_search_output(hits, gold_docids, snippet_chars):
    """Render retrieval hits as a ranked list, flagging gold documents."""
    lines = []
    for rank, hit in enumerate(hits, start=1):
        docid = str(hit.get("docid"))
        score = hit.get("score")
        mark = " ⭐" if docid in gold_docids else ""
        score_text = f" · score {score:.4f}" if isinstance(score, float) else ""
        snippet = _truncate(hit.get("snippet", ""), snippet_chars)
        lines.append(
            _details(
                f"<code>{rank}. docid {docid}</code>{score_text}{mark}",
                _fence(snippet),
            )
        )
    return "\n\n".join(lines) if lines else "_No results._"


def _format_tool_output(item, gold_docids, snippet_chars, document_chars):
    """Render a tool call's output, unpacking the shapes each tool returns."""
    raw = item.get("output")
    try:
        payload = json.loads(raw)
    except (TypeError, ValueError):
        return _fence(_truncate(str(raw), document_chars))

    if isinstance(payload, list):
        return _format_search_output(payload, gold_docids, snippet_chars)

    if isinstance(payload, dict) and "text" in payload:
        docid = str(payload.get("docid"))
        mark = " ⭐" if docid in gold_docids else ""
        body = _fence(_truncate(payload["text"], document_chars))
        return _details(f"<code>docid {docid}</code>{mark} — full document", body)

    if isinstance(payload, dict) and "error" in payload:
        return f"**Error:** {payload['error']}"

    return _fence(json.dumps(payload, indent=2, ensure_ascii=False), "json")


def _cell(value):
    """Make ``value`` safe inside a markdown table cell."""
    return str(value).replace("|", "\\|").replace("\n", " ")


def _format_metadata(record, query_id, gold_docids, evidence_docids, extra=None):
    """Build the header table: model settings, status, retrieval coverage, ``extra``."""
    metadata = record.get("metadata") or {}
    reasoning = metadata.get("reasoning") or {}
    counts = record.get("tool_call_counts") or {}
    retrieved = {str(d) for d in record.get("retrieved_docids") or []}

    rows = [
        ("Query ID", query_id),
        ("Model", metadata.get("model", "—")),
        ("Reasoning", f"effort={reasoning.get('effort', '—')}, summary={reasoning.get('summary', '—')}"),
        ("Status", record.get("status", "—")),
        ("Tool calls", ", ".join(f"{k}: {v}" for k, v in counts.items()) or "—"),
        ("Docs retrieved", len(retrieved)),
    ]
    for label, docids in (("Gold docs found", gold_docids), ("Evidence docs found", evidence_docids)):
        if docids:
            hit = sorted(docids & retrieved, key=str)
            rows.append((label, f"{len(hit)} / {len(docids)} ({', '.join(hit) if hit else 'none'})"))
    rows += list((extra or {}).items())

    lines = ["| Field | Value |", "| --- | --- |"]
    lines += [f"| {_cell(label)} | {_cell(value)} |" for label, value in rows]
    return "\n".join(lines)


def record_to_markdown(
    record,
    query_id=None,
    queries=None,
    golds=None,
    evidence=None,
    snippet_chars=SNIPPET_CHARS,
    document_chars=DOCUMENT_CHARS,
    extra=None,
):
    """Return the markdown transcript for a single run record.

    ``extra`` is an optional ``{label: value}`` mapping appended to the header table —
    useful for folding in eval results (judge verdict, predicted vs correct answer).
    """
    query_id = str(query_id if query_id is not None else record.get("query_id"))
    queries = queries if queries is not None else load_queries()
    golds = golds if golds is not None else load_qrels(GOLDS_FILE)
    evidence = evidence if evidence is not None else load_qrels(EVIDENCE_FILE)
    gold_docids = golds.get(query_id, set())
    evidence_docids = evidence.get(query_id, set())

    parts = [
        f"# Run — query {query_id}",
        _format_metadata(record, query_id, gold_docids, evidence_docids, extra),
    ]

    question = queries.get(query_id)
    if question:
        parts.append("## Query\n\n" + question)

    parts.append("## Trace")

    step = 0
    final_answers = []
    for item in record.get("result") or []:
        kind = item.get("type")
        if kind == "output_text":
            final_answers.append(str(item.get("output", "")).strip())
            continue

        step += 1
        if kind == "reasoning":
            output = item.get("output")
            blocks = output if isinstance(output, list) else [output]
            text = "\n\n---\n\n".join(str(b).strip() for b in blocks if b)
            parts.append(f"### Step {step} — Reasoning\n\n" + _details("reasoning", text))
        elif kind == "tool_call":
            tool_name = item.get("tool_name") or "unknown tool"
            body = [
                f"### Step {step} — Tool call: `{tool_name}`",
                "**Arguments**\n\n" + _fence(_format_arguments(item.get("arguments")), "json"),
                "**Output**\n\n"
                + _format_tool_output(item, gold_docids, snippet_chars, document_chars),
            ]
            parts.append("\n\n".join(body))
        else:
            parts.append(
                f"### Step {step} — {kind}\n\n"
                + _fence(json.dumps(item, indent=2, ensure_ascii=False), "json")
            )

    parts.append("## Final answer\n\n" + ("\n\n---\n\n".join(final_answers) or "_No final answer recorded._"))
    return "\n\n".join(parts) + "\n"


def run_to_markdown(query_id, runs_dir=DEFAULT_RUNS_DIR, **kwargs):
    """Return the markdown for ``query_id``, or ``None`` if there is no such run."""
    record = find_run(query_id, runs_dir)
    if record is None:
        return None
    return record_to_markdown(record, query_id=query_id, **kwargs)


def save_markdown(query_id, out_dir=DEFAULT_OUT_DIR, runs_dir=DEFAULT_RUNS_DIR, **kwargs):
    """Write ``markdown_runs/query_<id>.md`` for ``query_id`` and return its path.

    Raises ``KeyError`` if no run exists for the query.
    """
    markdown = run_to_markdown(query_id, runs_dir=runs_dir, **kwargs)
    if markdown is None:
        raise KeyError(f"no run found for query_id {query_id}")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"query_{query_id}.md"
    target.write_text(markdown)
    return target


def show_markdown(query_id, runs_dir=DEFAULT_RUNS_DIR, **kwargs):
    """Return the run for ``query_id`` as an IPython ``Markdown`` object (for notebooks)."""
    from IPython.display import Markdown

    markdown = run_to_markdown(query_id, runs_dir=runs_dir, **kwargs)
    if markdown is None:
        raise KeyError(f"no run found for query_id {query_id}")
    return Markdown(markdown)


def export_all(out_dir, runs_dir=DEFAULT_RUNS_DIR, **kwargs):
    """Write one markdown file per run into ``out_dir``; return the paths written."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tables = shared_tables()
    written = []
    for path, record in iter_runs(runs_dir):
        query_id = str(record.get("query_id") or path.stem)
        target = out_dir / f"query_{query_id}.md"
        target.write_text(record_to_markdown(record, query_id=query_id, **tables, **kwargs))
        written.append(target)
    return written


# --- PDFs -------------------------------------------------------------------

STYLESHEET = """
@page {
  size: A4;
  margin: 18mm 16mm 20mm 16mm;
  @bottom-center { content: counter(page) " / " counter(pages); font: 9pt sans-serif; color: #888; }
}
body { font: 10.5pt/1.5 "DejaVu Sans", sans-serif; color: #1a1a1a; }
h1 { font-size: 18pt; margin: 0 0 12pt; padding-bottom: 6pt; border-bottom: 2pt solid #1a1a1a;
     break-before: page; break-after: avoid; }
h1:first-of-type { break-before: avoid; }
h2 { font-size: 13pt; margin: 20pt 0 8pt; break-after: avoid; }
h3 { font-size: 11pt; margin: 14pt 0 6pt; color: #333; break-after: avoid; }
p { margin: 0 0 8pt; orphans: 2; widows: 2; }
table { border-collapse: collapse; width: 100%; margin: 0 0 12pt; font-size: 9.5pt; }
th, td { border: 0.5pt solid #ccc; padding: 3pt 6pt; text-align: left; vertical-align: top; }
th { background: #f2f2f2; }
tr { break-inside: avoid; }
code { font-family: "DejaVu Sans Mono", monospace; font-size: 8.5pt; }
pre { font-family: "DejaVu Sans Mono", monospace; font-size: 8pt; line-height: 1.35;
      background: #f7f7f7; border: 0.5pt solid #e0e0e0; border-radius: 2pt;
      padding: 5pt 7pt; margin: 0; white-space: pre-wrap; word-wrap: break-word; }
.fold { margin: 0 0 8pt; }
.fold-title { font-size: 9pt; font-weight: bold; color: #555; background: #ececec;
              padding: 3pt 6pt; border-radius: 2pt 2pt 0 0; break-after: avoid; }
.fold-title code { font-size: 9pt; }
.fold-body { border: 0.5pt solid #ececec; border-top: none; padding: 5pt 6pt; }
.fold-body pre { background: none; border: none; padding: 0; }
.subtitle { color: #666; margin-bottom: 18pt; }
"""

# The <details> markup emitted above, split into its three pieces.
_DETAILS_OPEN = re.compile(r"<details(?: open)?>\s*<summary>(.*?)</summary>", re.DOTALL)
_DETAILS_CLOSE = re.compile(r"</details>")


def _pdf_deps():
    """Import the PDF-only dependencies, which the markdown half must not require."""
    try:
        import markdown as markdown_lib
        from weasyprint import HTML
    except ImportError as exc:  # noqa: TRY003 - the install line is the whole point
        raise ImportError(
            "PDF export needs weasyprint and markdown: uv pip install weasyprint markdown"
        ) from exc
    return markdown_lib, HTML


def _unfold(text):
    """Turn ``<details>``/``<summary>`` into always-visible labelled sections.

    This rewrites the *markdown*, not the rendered HTML: python-markdown treats a
    ``<details>`` block as one opaque raw-HTML span and would pass the reasoning and
    code fences inside it through untouched. The ``markdown="1"`` attributes hand the
    contents back to the parser (via the ``md_in_html`` extension).
    """
    text = _DETAILS_OPEN.sub(
        lambda m: '<div class="fold" markdown="1">\n'
        f'<div class="fold-title" markdown="1">{m.group(1).strip()}</div>\n'
        '<div class="fold-body" markdown="1">',
        text,
    )
    return _DETAILS_CLOSE.sub("</div>\n</div>", text)


def markdown_to_html(text, title="Run transcript"):
    """Convert transcript markdown into a standalone, print-styled HTML document."""
    markdown_lib, _ = _pdf_deps()
    body = markdown_lib.markdown(
        _unfold(text),
        extensions=["tables", "fenced_code", "sane_lists", "md_in_html"],
    )
    return (
        "<!DOCTYPE html><html><head><meta charset='utf-8'>"
        f"<title>{title}</title><style>{STYLESHEET}</style></head>"
        f"<body>{body}</body></html>"
    )


def write_pdf(text, out_path, title="Run transcript"):
    """Render transcript markdown to a PDF at ``out_path``; return the path."""
    _, HTML = _pdf_deps()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    HTML(string=markdown_to_html(text, title=title)).write_pdf(out_path)
    return out_path


def run_to_pdf(query_id, out_dir=DEFAULT_PDF_DIR, runs_dir=DEFAULT_RUNS_DIR, tables=None, **kwargs):
    """Write ``query_<id>.pdf`` for one query and return its path.

    Raises ``KeyError`` if the query has no run.
    """
    kwargs.setdefault("snippet_chars", PDF_SNIPPET_CHARS)
    kwargs.setdefault("document_chars", PDF_DOCUMENT_CHARS)
    text = run_to_markdown(query_id, runs_dir=runs_dir, **(tables or shared_tables()), **kwargs)
    if text is None:
        raise KeyError(f"no run found for query_id {query_id}")
    return write_pdf(text, Path(out_dir) / f"query_{query_id}.pdf", title=f"Run — query {query_id}")


def _answer(value):
    """Display an answer, spelling out the blanks left by unparseable model output."""
    text = str(value).strip()
    return "_(no answer given)_" if text in ("", "nan", "None") else text


def _cover(rows, title):
    """Build a summary table introducing the runs bundled into a combined PDF."""
    lines = [
        f"# {title}",
        "",
        "<p class='subtitle'>One section per query: the question, what the model answered, "
        "and the full search-and-reasoning trace behind it.</p>",
        "",
        "| Query | Verdict | Predicted answer | Correct answer | Gold docs found |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['query_id']} | {row['verdict']} | {row['predicted']} | {row['correct']} | {row['gold']} |"
        )
    return "\n".join(lines)


def _gold_summary(text):
    """Pull the ``Gold docs found`` figure back out of a rendered header table."""
    match = re.search(r"\| Gold docs found \| ([^|(]*)", text)
    return match.group(1).strip() if match else "—"


def sample_frames(n, csv_path=EVAL_CSV, random_state=0):
    """Return ``{"passed": rows, "failed": rows}`` sampled from the judge results."""
    import pandas as pd

    df = pd.read_csv(csv_path)
    return {
        "passed": df[df["judge_correct"]].sample(n, random_state=random_state),
        "failed": df[~df["judge_correct"]].sample(n, random_state=random_state),
    }


def export_samples(n=2, out_dir=DEFAULT_PDF_DIR, csv_path=EVAL_CSV, combined=True, **kwargs):
    """Write ``n`` passed and ``n`` failed runs as PDFs; return the paths written.

    Each PDF carries the judge verdict and predicted vs. correct answer in its header
    table, so a run reads on its own. With ``combined``, also write ``samples.pdf``
    holding all of them behind a summary cover page.
    """
    kwargs.setdefault("snippet_chars", PDF_SNIPPET_CHARS)
    kwargs.setdefault("document_chars", PDF_DOCUMENT_CHARS)
    tables = shared_tables()
    out_dir = Path(out_dir)

    written, sections, cover_rows = [], [], []
    for label, rows in sample_frames(n, csv_path).items():
        for row in rows.itertuples():
            verdict = "correct" if row.judge_correct else "incorrect"
            extra = {
                "Judge verdict": verdict,
                "Predicted answer": _answer(row.predicted_answer),
                "Correct answer": _answer(row.correct_answer),
                "Judge confidence": row.confidence,
                "Citations": f"{row.num_citations} (recall {row.recall_positives:.2f}, "
                f"precision {row.precision_positives:.2f})",
            }
            text = run_to_markdown(row.query_id, **tables, extra=extra, **kwargs)
            if text is None:
                continue
            target = out_dir / label / f"query_{row.query_id}.pdf"
            written.append(write_pdf(text, target, title=f"Run — query {row.query_id} ({verdict})"))
            sections.append(text)
            cover_rows.append(
                {
                    "query_id": row.query_id,
                    "verdict": verdict,
                    "predicted": _answer(row.predicted_answer),
                    "correct": _answer(row.correct_answer),
                    "gold": _gold_summary(text),
                }
            )

    if combined and sections:
        title = f"Sample runs — {n} correct, {n} incorrect"
        bundle = "\n\n".join([_cover(cover_rows, title)] + sections)
        written.append(write_pdf(bundle, out_dir / "samples.pdf", title=title))
    return written


# --- CLI --------------------------------------------------------------------


def main(argv=None):
    import argparse
    import sys

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("query_ids", nargs="*", help="query ids to render")
    parser.add_argument("--pdf", action="store_true", help="render PDFs instead of markdown")
    parser.add_argument("--outline", action="store_true", help="list each run's step types instead of rendering")
    parser.add_argument("--all", action="store_true", help="render every run in the directory (markdown only)")
    parser.add_argument(
        "--samples", type=int, metavar="N", help="render N passed + N failed from the judge results (implies --pdf)"
    )
    parser.add_argument("--no-combined", action="store_true", help="with --samples, skip the bundled samples.pdf")
    parser.add_argument("--runs-dir", default=DEFAULT_RUNS_DIR, help="directory of run JSON files")
    parser.add_argument("-o", "--output", help="output file, or directory with --all/--pdf; default stdout")
    parser.add_argument("--snippet-chars", type=int, help="max chars per search snippet")
    parser.add_argument("--document-chars", type=int, help="max chars per fetched document")
    parser.add_argument("--full", action="store_true", help="never truncate tool output")
    args = parser.parse_args(argv)

    if args.outline:
        if not args.query_ids:
            parser.error("--outline needs at least one query_id")
        for query_id in args.query_ids:
            result = get_result(query_id, runs_dir=args.runs_dir)
            if result is None:
                print(f"{query_id}: not found")
                continue
            print(f"{query_id}: {len(result)} result items")
            for item in result:
                print(f"  - {item.get('type')} {item.get('tool_name') or ''}".rstrip())
        return 0

    pdf = args.pdf or args.samples is not None
    if pdf and args.all:
        parser.error("--all renders markdown only; use --samples N for a batch of PDFs")
    if not pdf and args.no_combined:
        parser.error("--no-combined only applies to --samples")

    # PDFs pay for every character in pages, so they truncate harder by default.
    snippet_default, document_default = (
        (PDF_SNIPPET_CHARS, PDF_DOCUMENT_CHARS) if pdf else (SNIPPET_CHARS, DOCUMENT_CHARS)
    )
    limits = {
        "snippet_chars": None if args.full else (args.snippet_chars or snippet_default),
        "document_chars": None if args.full else (args.document_chars or document_default),
    }

    if args.samples is not None:
        written = export_samples(
            args.samples, out_dir=args.output or DEFAULT_PDF_DIR, combined=not args.no_combined, **limits
        )
        for path in written:
            print(path)
        return 0 if written else 1

    if args.all:
        out_dir = args.output or DEFAULT_OUT_DIR
        written = export_all(out_dir, runs_dir=args.runs_dir, **limits)
        print(f"Wrote {len(written)} files to {out_dir}/", file=sys.stderr)
        return 0

    if not args.query_ids:
        parser.error("give at least one query_id, --all, or --samples N")

    tables = shared_tables()

    if pdf:
        written = []
        for query_id in args.query_ids:
            try:
                written.append(
                    run_to_pdf(
                        query_id,
                        out_dir=args.output or DEFAULT_PDF_DIR,
                        runs_dir=args.runs_dir,
                        tables=tables,
                        **limits,
                    )
                )
            except KeyError:
                print(f"{query_id}: not found", file=sys.stderr)
        for path in written:
            print(path)
        return 0 if written else 1

    sections, missing = [], []
    for query_id in args.query_ids:
        markdown = run_to_markdown(query_id, runs_dir=args.runs_dir, **tables, **limits)
        if markdown is None:
            missing.append(query_id)
        else:
            sections.append(markdown)

    for query_id in missing:
        print(f"{query_id}: not found", file=sys.stderr)
    if not sections:
        return 1

    text = "\n\n---\n\n".join(sections)
    if args.output:
        Path(args.output).write_text(text)
        print(f"Wrote {args.output}", file=sys.stderr)
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
