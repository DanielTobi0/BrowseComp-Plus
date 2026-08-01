"""Look up per-query run records by ``query_id``.

Each run under ``runs/<retriever>/<model>/`` is a single JSON file named after its
timestamp, so the ``query_id`` it belongs to is only visible inside the file. These
helpers scan a run directory and hand back the record (or just its ``result`` list)
for a given query.
"""

import json
from pathlib import Path

DEFAULT_RUNS_DIR = Path(__file__).parent / "runs" / "custom" / "qwen3.5-9b"


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


if __name__ == "__main__":
    import sys

    for qid in sys.argv[1:]:
        result = get_result(qid)
        if result is None:
            print(f"{qid}: not found")
        else:
            print(f"{qid}: {len(result)} result items")
            for item in result:
                print(f"  - {item.get('type')} {item.get('tool_name') or ''}".rstrip())
