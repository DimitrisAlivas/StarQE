"""
Data preprocessing and loading methods.

The preprocessing involves the following stages:

1. Using SPARQL formulas, the query_executor runs these queries from `formulas` against a GraphDB triple store.
   The results are stored in CSV format in directory `queries`.
2. The textual queries are converted to a binary format using the `converter` module.
3. The query dataset / data loader from the `loader` read these binary queries and take care of representing them as
   PyTorch Geometric graphs, with appropriate batch collation.
"""

from .loader import QueryGraphBatch

__all__ = [
    "QueryGraphBatch",
]
