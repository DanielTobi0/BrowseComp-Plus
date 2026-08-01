"""
Hybrid searcher: BM25 + FAISS dense retrieval fused via Reciprocal Rank Fusion,
then reranked with a Qwen3-Reranker cross-encoder.

Ported from the exploratory version in note.ipynb into a proper BaseSearcher so
it can be used from the CLI (`--searcher-type custom`) and MCP server like any
other retriever.
"""

import logging
from argparse import Namespace
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .base import BaseSearcher
from .bm25_searcher import BM25Searcher
from .faiss_searcher import GPU_INFERENCE_LOCK, FaissSearcher

logger = logging.getLogger(__name__)


class HybridSearcher(BaseSearcher):
    @classmethod
    def parse_args(cls, parser):
        # BM25 side
        parser.add_argument(
            "--bm25-index-path",
            required=True,
            help="Path to the Lucene BM25 index (e.g. indexes/bm25).",
        )

        # FAISS / dense side
        parser.add_argument(
            "--faiss-index-path",
            required=True,
            help="Glob pattern for FAISS pickle shards (e.g. indexes/qwen3-embedding-8b/corpus.shard*_of_4.pkl).",
        )
        parser.add_argument(
            "--embedding-model-name",
            default="Qwen/Qwen3-Embedding-8B",
            help="Embedding model name for dense retrieval (default: %(default)s)",
        )
        parser.add_argument(
            "--embedding-normalize",
            action="store_true",
            default=True,
            help="Whether to normalize dense embeddings (default: %(default)s)",
        )
        parser.add_argument(
            "--embedding-pooling",
            default="eos",
            help="Pooling method for the embedding model (default: %(default)s)",
        )
        parser.add_argument(
            "--embedding-torch-dtype",
            default="float16",
            choices=["float16", "bfloat16", "float32"],
            help="Torch dtype for the embedding model (default: %(default)s)",
        )
        parser.add_argument(
            "--embedding-task-prefix",
            default=(
                "Instruct: Given a web search query, retrieve relevant passages "
                "that answer the query\nQuery:"
            ),
            help="Task prefix prepended to queries for the embedding model.",
        )
        parser.add_argument(
            "--embedding-max-length",
            type=int,
            default=8192,
            help="Maximum sequence length for the embedding model (default: %(default)s)",
        )
        parser.add_argument(
            "--dataset-name",
            default="Tevatron/browsecomp-plus-corpus",
            help="Dataset used to resolve full document text (default: %(default)s)",
        )
        parser.add_argument(
            "--faiss-gpu",
            action="store_true",
            default=False,
            help=(
                "Keep the FAISS index on GPU. Off by default: a hybrid searcher "
                "already keeps the embedding model + reranker on GPU, and cloning "
                "the FAISS index there too competes for VRAM."
            ),
        )

        # Fusion
        parser.add_argument(
            "--retrieve-k",
            type=int,
            default=20,
            help="Number of candidates each retriever contributes before fusion (default: %(default)s)",
        )
        parser.add_argument(
            "--rrf-k",
            type=int,
            default=60,
            help="RRF constant k (default: %(default)s)",
        )

        # Reranking
        parser.add_argument(
            "--no-rerank",
            action="store_true",
            default=False,
            help="Skip cross-encoder reranking; return RRF-fused results directly.",
        )
        parser.add_argument(
            "--reranker-model-name",
            default="Qwen/Qwen3-Reranker-0.6B",
            help="Reranker model name (default: %(default)s)",
        )
        parser.add_argument(
            "--reranker-max-length",
            type=int,
            default=8192,
            help="Maximum sequence length for the reranker (default: %(default)s)",
        )
        parser.add_argument(
            "--reranker-batch-size",
            type=int,
            default=4,
            help="Mini-batch size for reranking (default: %(default)s)",
        )
        parser.add_argument(
            "--reranker-task",
            default="Given a web search query, retrieve relevant passages that answer the query",
            help="Instruction describing the retrieval task, shown to the reranker.",
        )

    def __init__(self, args):
        self.args = args

        logger.info("Initializing hybrid searcher (BM25 + FAISS)...")

        self.bm25_searcher = BM25Searcher(Namespace(index_path=args.bm25_index_path))

        if args.faiss_gpu:
            self.faiss_searcher = FaissSearcher(
                Namespace(
                    index_path=args.faiss_index_path,
                    model_name=args.embedding_model_name,
                    normalize=args.embedding_normalize,
                    pooling=args.embedding_pooling,
                    torch_dtype=args.embedding_torch_dtype,
                    dataset_name=args.dataset_name,
                    task_prefix=args.embedding_task_prefix,
                    max_length=args.embedding_max_length,
                )
            )
        else:
            self.faiss_searcher = self._build_faiss_searcher_cpu(args)

        self.reranker_tokenizer = None
        self.reranker_model = None
        self._reranker_token_true_id = None
        self._reranker_token_false_id = None
        self._reranker_prefix_tokens = None
        self._reranker_suffix_tokens = None
        if not args.no_rerank:
            self._load_reranker()

        logger.info("Hybrid searcher initialized successfully")

    @staticmethod
    def _build_faiss_searcher_cpu(args) -> FaissSearcher:
        # Keep the FAISS index on CPU: cloning it onto the GPU (plus its scratch
        # workspace) competes with the embedding model + reranker for VRAM.
        import faiss

        original_get_num_gpus = faiss.get_num_gpus
        faiss.get_num_gpus = lambda: 0
        try:
            return FaissSearcher(
                Namespace(
                    index_path=args.faiss_index_path,
                    model_name=args.embedding_model_name,
                    normalize=args.embedding_normalize,
                    pooling=args.embedding_pooling,
                    torch_dtype=args.embedding_torch_dtype,
                    dataset_name=args.dataset_name,
                    task_prefix=args.embedding_task_prefix,
                    max_length=args.embedding_max_length,
                )
            )
        finally:
            faiss.get_num_gpus = original_get_num_gpus

    def _load_reranker(self) -> None:
        logger.info(f"Loading reranker: {self.args.reranker_model_name}")

        self.reranker_tokenizer = AutoTokenizer.from_pretrained(
            self.args.reranker_model_name, padding_side="left"
        )
        self.reranker_model = (
            AutoModelForCausalLM.from_pretrained(
                self.args.reranker_model_name, torch_dtype=torch.float16
            )
            .to("cuda" if torch.cuda.is_available() else "cpu")
            .eval()
        )

        self._reranker_token_false_id = self.reranker_tokenizer.convert_tokens_to_ids("no")
        self._reranker_token_true_id = self.reranker_tokenizer.convert_tokens_to_ids("yes")

        prefix = (
            "<|im_start|>system\nJudge whether the Document meets the requirements based on "
            'the Query and the Instruct provided. Note that the answer can only be "yes" or "no".'
            "<|im_end|>\n<|im_start|>user\n"
        )
        suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        self._reranker_prefix_tokens = self.reranker_tokenizer.encode(prefix, add_special_tokens=False)
        self._reranker_suffix_tokens = self.reranker_tokenizer.encode(suffix, add_special_tokens=False)

        logger.info("Reranker loaded successfully")

    @staticmethod
    def _reciprocal_rank_fusion(
        result_lists: List[List[Dict[str, Any]]], rrf_k: int = 60
    ) -> List[Dict[str, Any]]:
        """Fuse multiple ranked result lists (each: list of {docid, score, text}) via RRF."""
        fused_scores: Dict[str, float] = {}
        doc_cache: Dict[str, str] = {}

        for results in result_lists:
            for rank, hit in enumerate(results, start=1):
                docid = hit["docid"]
                fused_scores[docid] = fused_scores.get(docid, 0.0) + 1.0 / (rrf_k + rank)
                doc_cache.setdefault(docid, hit["text"])

        fused = [
            {"docid": docid, "score": score, "text": doc_cache[docid]}
            for docid, score in fused_scores.items()
        ]
        fused.sort(key=lambda x: x["score"], reverse=True)
        return fused

    def _format_reranker_pair(self, query: str, doc_text: str) -> str:
        return f"<Instruct>: {self.args.reranker_task}\n<Query>: {query}\n<Document>: {doc_text}"

    @torch.no_grad()
    def _score_batch(self, query: str, batch: List[Dict[str, Any]]) -> List[float]:
        pairs = [self._format_reranker_pair(query, c["text"]) for c in batch]

        budget = (
            self.args.reranker_max_length
            - len(self._reranker_prefix_tokens)
            - len(self._reranker_suffix_tokens)
        )
        tokenized = self.reranker_tokenizer(
            pairs, padding=False, truncation="longest_first", return_attention_mask=False, max_length=budget
        )
        for i, ids in enumerate(tokenized["input_ids"]):
            tokenized["input_ids"][i] = self._reranker_prefix_tokens + ids + self._reranker_suffix_tokens

        inputs = self.reranker_tokenizer.pad(
            tokenized, padding=True, return_tensors="pt", max_length=self.args.reranker_max_length
        )
        inputs = {k: v.to(self.reranker_model.device) for k, v in inputs.items()}

        # logits_to_keep=1: only the last position's logits are needed, and computing
        # logits for the full sequence (the model's default) can spike memory by
        # several GiB per batch once prompts approach reranker_max_length tokens.
        with GPU_INFERENCE_LOCK:
            logits = self.reranker_model(**inputs, logits_to_keep=1).logits[:, -1, :]
        true_scores = logits[:, self._reranker_token_true_id]
        false_scores = logits[:, self._reranker_token_false_id]
        log_probs = torch.nn.functional.log_softmax(
            torch.stack([false_scores, true_scores], dim=1), dim=1
        )
        return log_probs[:, 1].exp().tolist()

    def _rerank(self, query: str, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Score each candidate against the query with the reranker.

        Runs in mini-batches rather than one big batch: with long documents padded
        or truncated up to reranker_max_length tokens, scoring all candidates at
        once can spike activation memory enough to OOM alongside the embedding model.
        """
        if not candidates:
            return []

        batch_size = self.args.reranker_batch_size
        all_scores: List[float] = []
        for start in range(0, len(candidates), batch_size):
            batch = candidates[start : start + batch_size]
            all_scores.extend(self._score_batch(query, batch))
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        reranked = [{**c, "score": s} for c, s in zip(candidates, all_scores)]
        reranked.sort(key=lambda x: x["score"], reverse=True)
        return reranked

    def search(self, query: str, k: int = 10) -> List[Dict[str, Any]]:
        retrieve_k = max(self.args.retrieve_k, k)
        bm25_results = self.bm25_searcher.search(query, k=retrieve_k)
        dense_results = self.faiss_searcher.search(query, k=retrieve_k)
        fused = self._reciprocal_rank_fusion(
            [bm25_results, dense_results], rrf_k=self.args.rrf_k
        )

        if self.args.no_rerank:
            return fused[:k]

        return self._rerank(query, fused)[:k]

    def get_document(self, docid: str) -> Optional[Dict[str, Any]]:
        return self.faiss_searcher.get_document(docid)

    @property
    def search_type(self) -> str:
        return "Hybrid (BM25 + FAISS, RRF fusion + rerank)"

    def search_description(self, k: int = 10) -> str:
        return (
            f"Perform a hybrid search (BM25 + dense retrieval, fused via Reciprocal Rank "
            f"Fusion{'' if self.args.no_rerank else ' and reranked with a cross-encoder'}) "
            f"on a knowledge source. Returns top-{k} hits with docid, score, and snippet. "
            "The snippet contains the document's contents (may be truncated based on token limits)."
        )

    def get_document_description(self) -> str:
        return "Retrieve a full document by its docid."


# Backwards-compatible alias: docs/custom_retriever.md refers to this as "your
# custom searcher", plugged in via `--searcher-type custom`.
CustomSearcher = HybridSearcher
