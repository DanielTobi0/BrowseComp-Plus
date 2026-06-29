"""
Searchers package for different search implementations.
"""

from enum import Enum

from .base import BaseSearcher
from .bm25_searcher import BM25Searcher
from .custom_searcher import CustomSearcher


def _get_faiss_searcher():
    from .faiss_searcher import FaissSearcher
    return FaissSearcher


def _get_reasonir_searcher():
    from .faiss_searcher import ReasonIrSearcher
    return ReasonIrSearcher


class SearcherType(Enum):
    """Enum for managing available searcher types and their CLI mappings."""

    BM25 = ("bm25", BM25Searcher)
    FAISS = ("faiss", _get_faiss_searcher)
    REASONIR = ("reasonir", _get_reasonir_searcher)
    CUSTOM = (
        "custom",
        CustomSearcher,
    )  # Your custom searcher class, yet to be implemented

    def __init__(self, cli_name, searcher_class):
        self.cli_name = cli_name
        self.searcher_class = searcher_class

    @classmethod
    def get_choices(cls):
        """Get list of CLI choices for argument parser."""
        return [searcher_type.cli_name for searcher_type in cls]

    @classmethod
    def get_searcher_class(cls, cli_name):
        """Get searcher class by CLI name."""
        for searcher_type in cls:
            if searcher_type.cli_name == cli_name:
                cls_or_loader = searcher_type.searcher_class
                # Call lazy loader functions on demand
                if callable(cls_or_loader) and not isinstance(cls_or_loader, type):
                    return cls_or_loader()
                return cls_or_loader
        raise ValueError(f"Unknown searcher type: {cli_name}")


__all__ = ["BaseSearcher", "SearcherType"]
