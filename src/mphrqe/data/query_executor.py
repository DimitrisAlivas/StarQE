"""Perform SPARQL queries to generate the dataset."""
import csv
import gzip
import hashlib
import json
import logging
import random
import re
import traceback
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from requests import HTTPError

from .config import formula_root, query_root, sparql_endpoint_address as default_sparql_endpoint_address, sparql_endpoint_options as default_sparql_endpoint_options

__all__ = [
    "execute_queries",
]

logger = logging.getLogger(__name__)

# This is the random seed used to shuffle the queries before storing them. This simplifies the later sampling from sampling to taking the top-k
shuffling_random_seed = 58148615010101


def execute_queries(
    source_directory: Optional[Path] = None,
    target_directory: Optional[Path] = None,
    sparql_endpoint: Optional[str] = None,
    sparql_endpoint_options: Optional[Dict[str, Any]] = None,
    continue_on_error: bool = False,
) -> None:
    """
    Perform the higher order queries in all subdirectories and store their results in CSV files.

    The output of these higher order queries is such that they are themselves queries.
    to achieve this, the variables of the higher order queries must be as follows:

    * ?sN the subject of the Nth triple in the query
    * ?pN the predicate of the Nth triple
    * ?oN the object of the Nth triple
    * ?qrNiM the Mth qualifier relation of the Nth triple
    * ?qvNiM the Mth qualifier value of the Nth triple
    * ?diameter The diameter of the query
    * ?XN_targets the answers to the query, X indicates whether this is a subject, predicate, object, qualifier relation or value, N is the index of the triple
        * If this is a variable used in multiple triples, the it is just as normal, but with _targets appended. for example in a 2i query, the target would be ?o1_o2_targets

    Known values in the query must be their URL. The N-th variable (all indices 0 indexed!) must be indicated with "?varN".
    For the indices on the qualifiers, N refers to the triple it belongs to and M is the index of the qulifier on that triple
    Columns which always have the same values MUST be joined by joining the column names by _
    Columns which represent the same variable MUST be joined by joining the column names by _
    Note that this joining rule *also* applies if the common value is between a qv and a subject or object of a triple with more information about the qv.
    These joins are used to verify that the graph is connected.

    For example, a 2 hop query with 1 qualifier on each edge would heave the following variables:

    s0,p0,o0_s1_var,qr0i0,qv0i0,p1,qr1i0,qv1i0,diameter,o1_targets


    In the columns is either a URI, or multiple URIs separated by "|" for the targets

    For this concrete query, with answers wd:Q4 and wd:Q6
    select ?target
    <<wd:Q1 p:P1 ?var1>> p:P2 wd:Q3
    <<?var1 p:P2 ?target>> p:P3 wd:Q5

    The data in the CSV would be

    wd:Q1,p:P1,?var1,p:P2,wd:Q3,p:P2,p:P3,wd:Q5,2,wd:Q4|wd:Q6

    (note: prefixes should be expanded.)

    The hierarchical structure will be preserved, but for each file in the source directory.


    Parameters
    ----------
    source_directory: Optional[Path] = None
        The root directory containing the queries to be executed, all queries (in files ending in .sparql) recursively within this directory will be executed as formulas.

    target_directory: Optional[Path] = None
        The root directory in which the outcomes of the formulas is to be stored. the results are stored in the same relative path where the .sparql file was found.
        the outcome is stored as a gzipped .csv file

    sparql_endpoint: Optional[str] = None
        The address of the sparql endpoint

    sparql_endpoint_options: Optional[Dict[str, Any]] = None
        Options to be passed to the SPARQL endpoint upon connecting. This could include things like authentication info, etc.

    continue_on_error: bool = False
        If a query fails and continue_on_error is True, this prints a stacktrace and continues with other queries.
        If a query fails and continue_on_error is False, an Exception is raised.

    """
    source_directory = source_directory or formula_root
    target_directory = target_directory or query_root
    sparql_endpoint = sparql_endpoint or default_sparql_endpoint_address
    sparql_endpoint_options = sparql_endpoint_options or default_sparql_endpoint_options

    if not list(source_directory.rglob("*.sparql")):
        logger.warning(f"Empty source directory: {source_directory.as_uri()}")

    for query_file_path in source_directory.rglob("*.sparql"):
        query = query_file_path.read_text()
        new_query_hash = hashlib.md5(query.encode()).hexdigest()

        name_stem = query_file_path.stem
        # TODO: Code duplication to converter.py

        # get absolute source path
        relative_source_path = query_file_path.relative_to(source_directory)
        relative_source_directory = relative_source_path.parent

        # compute the destination path
        absolute_target_directory = target_directory.joinpath(relative_source_directory)
        absolute_target_directory.mkdir(parents=True, exist_ok=True)
        absolute_target_path = absolute_target_directory.joinpath(name_stem).with_suffix(".csv.gz").resolve()
        logger.info(f"{query_file_path.as_uri()} -> {absolute_target_path.as_uri()}")

        # the status file
        status_file_path = absolute_target_directory.joinpath(name_stem + "_stats").with_suffix(".json")

        # we only need to do the querying if the query has updated or there is no such file
        if absolute_target_path.is_file():
            # check the old hash from the stats file if it exists
            if status_file_path.is_file():
                with status_file_path.open("r") as status_file:
                    old_stats = json.load(status_file)
                old_query_hash = old_stats["hash"]
                if old_query_hash == new_query_hash:
                    logger.info(f"Queries already exist for {query_file_path.as_uri()} and hash matches. Not performing the query.")
                    continue
                else:
                    logger.warning(f"Queries exist for {query_file_path.as_uri()}, but hash does not match. Removing and regenerating!")
            else:
                logger.warning(f"Queries exist for {query_file_path.as_uri()}, but no stats file. Removing and regenerating!")
            # an old version exists but hashes do not match, we remove it, the user was warned above
            absolute_target_path.unlink()

        # perform the query and store the results
        logger.warning(f"Performing query for {query_file_path.as_uri()} to {absolute_target_path.as_uri()}")
        try:
            try:
                count = _execute_one_query(query, absolute_target_path, sparql_endpoint=sparql_endpoint, sparql_endpoint_options=sparql_endpoint_options)
            except HTTPError as e:
                raise Exception(str(e) + str(e.response.content)) from e
            new_stats = {"name": name_stem, "hash": new_query_hash, "count": count}
            try:
                with status_file_path.open("w") as status_file:
                    json.dump(new_stats, status_file)
            except Exception:
                # something went wrong writing the stats file, best to remove it and crash.
                logger.error("Failed writing the stats, removing the file to avoid inconsistent state")
                if status_file_path.exists():
                    status_file_path.unlink()
                raise
        except Exception as err:
            logging.error("Something went wrong executing the query, removing the output file")
            if absolute_target_path.exists():
                absolute_target_path.unlink()
            if continue_on_error:
                traceback.print_tb(err.__traceback__)
            else:
                raise


def _execute_one_query(query: str, destination_path: Path, sparql_endpoint: str, sparql_endpoint_options: Dict[str, Any]) -> int:
    """
    Performs the query provided and writes the results as a CSV to the destination.
    the queries are shuffled randomly (fixed seed) before storing to make later sampling a top-k operation instead of real sampling
    """
    from rdflib.plugins.stores.sparqlstore import SPARQLStore
    store = SPARQLStore(sparql_endpoint, returnFormat="csv", method="POST", **sparql_endpoint_options)  # headers={}
    global_logger = logging.getLogger()
    original_level = global_logger.getEffectiveLevel()
    global_logger.setLevel(logging.ERROR)
    try:
        result = store.query(query)
    finally:
        global_logger.setLevel(original_level)

    # convert to string and take of the leading '?'
    fieldnames = [
        var.toPython()[1:]
        for var in result.vars
    ]
    assert_query_validity(fieldnames)
    all_queries = list(result)
    query_count = len(all_queries)

    # shuffle all the answers
    r = random.Random(shuffling_random_seed)
    r.shuffle(all_queries)
    with gzip.open(destination_path, compresslevel=6, mode="wt", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames, extrasaction="raise", dialect="unix", quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()
        for one_query in all_queries:
            writer.writerow(one_query.asdict())
    return query_count


subject_matcher = re.compile("s[0-9]+")
predicate_matcher = re.compile("p[0-9]+")
object_matcher = re.compile("o[0-9]+")
qualifier_relation_matcher = re.compile("qr[0-9]+i[0-9]+")
qualifier_value_matcher = re.compile("qv[0-9]+i[0-9]+")


def assert_query_validity(fieldnames: Iterable[str]) -> bool:
    """
    Some heuristics to check whether the headers make sense for a query.
    This is not exhaustive, some specific cases can still pass this test.
    In particular cases where self loops not connected to the rest of the graph go undetected.
    """
    # We need a modifiable list
    fieldnames = list(fieldnames)
    # quick check for duplicates
    allsubparts = [subpart for field in fieldnames for subpart in field.split("_") if not (subpart == "var" or subpart == "target")]
    duplicates = set([x for x in allsubparts if allsubparts.count(x) > 1])
    assert len(duplicates) == 0, "The following specifications where found in more than one specification: " + str(duplicates)

    expected_triple_count = sum([1 for fields in fieldnames for subpart in fields.split('_') if subject_matcher.match(subpart)])
    assert expected_triple_count > 0, "At least one triple must be completely specififies with subject, predicate and object"
    expected_qualifier_count = sum([1 for fields in fieldnames for subpart in fields.split('_') if qualifier_relation_matcher.match(subpart)])
    # keep track of what has been found
    target_found = False
    diameter_found = False
    spo_found = [[False, False, False] for i in range(expected_triple_count)]
    # The qualifier relation and value must refer to the same triple, we remmeber the triple number and verify in the end.
    qr_qv_found = [[-1, -1] for i in range(expected_qualifier_count)]
    for part in fieldnames:
        if part.endswith("_target"):
            assert target_found is False, "more than one column with _target found. Giving up."
            target_found = True
            targetHeader = part[:-len("_target")]
            subparts = targetHeader.split('_')
            assert "var" not in subparts, "'var' cannot occur in the same header with 'target', , got {part}"
        if part.endswith("_var"):
            varheader = part[:-len("_var")]
            subparts = varheader.split("_")
            assert "target" not in subparts, f"'var' cannot occur in the same header with 'target', got {part}"
        if part.endswith("_target") or part.endswith("_var"):
            # This header must contain only s, o, qr, qv
            for subpart in subparts:
                if not any((
                    subject_matcher.match(subpart),
                    object_matcher.match(subpart),
                    qualifier_relation_matcher.match(subpart),
                    qualifier_value_matcher.match(subpart),
                )):
                    raise ValueError(
                        f"the target header can only contain subject, predicate, object, query relation, or query "
                        f"value, contained {subpart}",
                    )
        else:
            # split the part if it is used multiple times
            subparts = part.split("_")
        assert not len(subparts) == 0, "Did you create a column with a header without any actual fields in it?"
        if len(subparts) == 1:
            if subparts[0] == "diameter":
                assert not diameter_found, "diameter specified more than once"
                diameter_found = True
                continue
        elif len(subparts) > 1:
            # there can be no predicates, qualifier_relations and diameter here
            for subpart in subparts:
                if subpart.startswith('p') or subpart.startswith('qr') or subpart == 'diameter':
                    logging.warning(f"""Found a joined header {part} with a part that is typically not joined {subpart}.
                    This might be intended, but is strange. Perhaps a case  where you want to do something with a shared edge?""")
        for subpart in subparts:
            # we treat each different: subject, predicate, object, qr, qv
            if subject_matcher.match(subpart):
                # it is a subject
                tripleIndex = int(subpart[1:])
                assert tripleIndex < expected_triple_count, \
                    f"Found a {subpart} refering to triple {tripleIndex} while we only have {expected_triple_count} triples"
                assert not spo_found[tripleIndex][0]
                spo_found[tripleIndex][0] = True
            elif predicate_matcher.match(subpart):
                tripleIndex = int(subpart[1:])
                assert tripleIndex < expected_triple_count, \
                    f"Found a {subpart} refering to triple {tripleIndex} while we only have {expected_triple_count} triples"
                assert not spo_found[tripleIndex][1]
                spo_found[tripleIndex][1] = True
            elif object_matcher.match(subpart):
                tripleIndex = int(subpart[1:])
                assert tripleIndex < expected_triple_count, \
                    f"Found a {subpart} refering to triple {tripleIndex} while we only have {expected_triple_count} triples"
                assert not spo_found[tripleIndex][2]
                spo_found[tripleIndex][2] = True
            elif qualifier_relation_matcher.match(subpart):
                indices = subpart[2:].split('i')
                tripleIndex = int(indices[0])
                assert tripleIndex < expected_triple_count, f"qualifier relation {subpart} refers to non existing triple {tripleIndex}"
                qualIndex = int(indices[1])
                assert qr_qv_found[qualIndex][0] == -1, f"qualifier relation qrXi{qualIndex} set twice"
                qr_qv_found[qualIndex][0] = tripleIndex
            elif qualifier_value_matcher.match(subpart):
                indices = subpart[2:].split('i')
                tripleIndex = int(indices[0])
                assert tripleIndex < expected_triple_count, f"qualifier value {subpart} refers to non existing triple {tripleIndex}"
                qualIndex = int(indices[1])
                assert qualIndex < expected_qualifier_count, f"Found qualifier value {subpart} for which there is no corresponding qr{qualIndex}"
                assert qr_qv_found[qualIndex][1] == -1, f"qualifier value qvXi{qualIndex} set twice"
                qr_qv_found[qualIndex][1] = tripleIndex
            else:
                raise AssertionError(f"Unknown column with name '{subpart}'")
    # The qualifier relation and value must refer to the same triple, we remmeber the triple number and verify in the end.
    assert target_found, "No column found with _target"
    assert diameter_found, "No query diameter specified"
    for (index, spo) in enumerate(spo_found):
        assert spo[0], f"Could not find subject s{index}"
        assert spo[1], f"Could not find predicate p{index}"
        assert spo[2], f"Could not find object o{index}"
    for (index, qr_qv) in enumerate(qr_qv_found):
        assert qr_qv[0] != -1, f"Could not find qrXi{index}"
        assert qr_qv[1] != -1, f"Could not find qvXi{index}"
    # Finally, some more checks to catch at least some cases where the query graph is not connected. For a one triple query we are done here
    if expected_triple_count == 1:
        return True
    # For each triple there must be at least one end that is joined with an s or o of another triple or with a qv.
    # We only check that it at least co-occurs with something. This leaves some case where triples are looping on themselves undiscovered.
    # Note that at this point we are already sure that all qualifiers are connected to triples and all fields have already been checked above.
    # In principle, this loop could occur as part of the above code checking the field, but that would render it pretty much unreadble.
    connectionFound = [False for i in range(expected_triple_count)]
    for part in fieldnames:
        if part.endswith("_target"):
            targetHeader = part[:-len("_target")]
            subparts = targetHeader.split('_')
        elif part.endswith("_var"):
            varheader = part[:-len("_var")]
            subparts = varheader.split("_")
        else:
            # split the part if it is used multiple times
            subparts = part.split("_")
        if len(subparts) == 1:
            # single fields do not give us any information about connectedness
            continue
        for subpart in subparts:
            # we treat each different: subject, predicate, object, qr, qv.
            if subject_matcher.match(subpart):
                tripleIndex = int(subpart[1:])
                connectionFound[tripleIndex] = True
            elif object_matcher.match(subpart):
                tripleIndex = int(subpart[1:])
                connectionFound[tripleIndex] = True
            else:
                # Just fine. Other things could occur in these fields, but we can ignore these here.
                pass
    assert all(connectionFound), "It seems like not all triples in the query are connected to another triple or qualifier"
    return True
