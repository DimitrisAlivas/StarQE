"""Data path configuration."""
import logging
import pathlib

__all__ = [
    "formula_root",
    "mapping_root",
    "query_root",
    "binary_query_root",
    "triples_root",
    "sparql_endpoint_address",
    "sparql_endpoint_options",
]

root = pathlib.Path(__file__, "..", "..", "..", "..").resolve()
# fixme: they are the same?
old_data_root = root.joinpath("wd50k_original_data")
data_root = root.joinpath("data")

formula_root = data_root.joinpath("formulas")
mapping_root = data_root.joinpath("mappings")
query_root = data_root.joinpath("queries")
binary_query_root = data_root.joinpath("binaryQueries")
triples_root = data_root.joinpath("triple_data")
anzograph_init_root = data_root.joinpath("anzograph")

sparql_endpoint_address = "http://localhost:7200/repositories/wd50K"
# sparql_endpoint_address = "https://localhost:8256/sparql"
sparql_endpoint_options = {'verify': False}

# TODO: Should this be done here?
logging.captureWarnings(True)
