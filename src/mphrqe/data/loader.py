"""
This module implements the loading of queries into a PyTorch Dataset.
The user can select the parts of the data to be loaded in test, training, and validation.
Besides, it allows the user to decide how many elements to take from each part of these sets.
"""
import dataclasses
import itertools
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence, Tuple, Union

import torch
from torch.utils.data import DataLoader, Dataset

from .converter import QueryData, TensorBuilder
from .generated.query_pb2 import QueryData as pb_QueryData
from .mapping import get_entity_mapper, get_relation_mapper
from ..typing import LongTensor

__all__ = [
    "get_query_data_loaders",
    "resolve_sample",
    "QueryGraphBatch",
]

logger = logging.getLogger(__name__)


class Sample:
    """
    A sample is a part of a split.

    You can specify which part of the data to be included and how many you want from this part.
    The final amounts will be sampled proportionally from the different matching data files.

    Parameters
        ----------
        selector : str
            The selector allows simple glob operations,
            for example `"/**/1qual/"` will select queries with one qualifier.
        amount : int, str, or a callable
            How many elements to sample from the sets indicated.
            This is specified by either:
                * a number indicating the amount
                * the string "*" indicating all
                * a callable that, given the total number available returns how many to sample.
            The latter can be used to control the amount depending on what is in the dataset,
               e.g., at most 10000  `lambda available: min(10000, available)` or
               10% of the data `lambda available: int(0.10 * available)`
        reify: bool, default=False
            Should the query be reified during the load?
            If True, triples which have qualifiers will be reified and an extra variable node will be added
            to represent the blank node. Then, for each qualifier a respective statement will be added with the
            respective node as the subject.In effect, the queries will have explicit qualifiers.

            This option conflicts with remove_qualifiers. Only one of these two can be True
        remove_qualifiers: bool, defaul = False
            Must qaulifiers be removed from the queries? If True, the qualifiers tensor will not contain anything.

            This option conflicts with reify. Only one of these two can be True

    """

    def __init__(self, selector: str, amount: Union[int, str, Callable[[int], int]], reify: bool = False, remove_qualifiers: bool = False):
        assert callable(amount) or (isinstance(amount, int) and amount >= 0) or amount == "*", \
            f"Illegal amount specification, got {amount}. Check the documentation."
        assert not (remove_qualifiers and reify), "Asked to both reify and remove qualifiers, this is almost certainly a mistake."
        self.selector = selector
        if isinstance(amount, int):
            def exact(available):
                assert available >= amount, \
                    f"Not enough data available. Requested {amount}, but there is only {available}"
                return amount

            self._amount = exact
        elif amount == "*":
            self._amount = lambda available: available
        else:
            self._amount = amount  # type: ignore
        self.reify = reify
        self.remove_qualifiers = remove_qualifiers

    def amount(self, x: int) -> int:
        return self._amount(x)


@dataclasses.dataclass()
class DatafileInfo:
    """Stores information about one specific file from which data is loaded"""
    source_file: Path
    amount_available: int
    amount_requested: int
    hashcode: str


def get_query_datasets(
    data_root: Path,
    train: Iterable[Sample],
    validation: Iterable[Sample],
    test: Iterable[Sample],
) -> Tuple[Mapping[str, Dataset], Mapping[str, Sequence[Mapping]]]:
    """
    This code is used to load datasets for training, validating and testing a model.

    For each of train, test, and validation, a list of Sample is expected,


    Note, the implementations does not check for overlap.
    Be careful that you do not accidentally load the same set twice.

    If the selector selects multiple datasets, the sampling will be uniform,
    taking into account the size of the datasets.
    If you want to select all items in a dataset, specify QuerySetLoader.ALL
    """
    data_root = Path(data_root).expanduser().resolve()
    logger.info(f"Using data_root={data_root.as_uri()}")

    # the procedure is the same for each of the splits. Only the splits selected are different.
    datasets = dict()
    # store information about the actually loaded data.
    information = defaultdict(list)
    for split_name, split in dict(
        train=train,
        validation=validation,
        test=test,
    ).items():
        # collect all samples
        all_samples = []
        for sample in split:
            total_available = 0
            datafiles_and_info = []
            glob_pattern = "./" + sample.selector + "/**/*" + split_name + "_stats.json"
            stat_files = list(data_root.glob(glob_pattern))
            assert len(stat_files) > 0, f"The number of files for split {split_name} and pattern {sample.selector} was empty. This is close to always a mistake in the selector."
            for stats_file_path in stat_files:
                # for sample_path in data_root.glob("./" + sample.selector + "/*/"):
                #    for stats_file_path in sample_path.glob("*" + splitname + "_stats.json"):
                with open(stats_file_path) as stats_file:
                    stats = json.load(stats_file)
                amount_in_file = stats["count"]
                total_available += amount_in_file
                setname = stats["name"]
                # the setname already contains the splitname!
                data_file = stats_file_path.parents[0] / (setname + ".proto")
                assert data_file.exists(), \
                    f"The datafile {data_file} refered from {stats_file_path} could not be found."
                datafiles_and_info.append(DatafileInfo(data_file, amount_in_file, 0, stats["hash"]))

            # get how many we need, and then split proportionally
            requested_amount = sample.amount(total_available)
            fraction = requested_amount / total_available

            still_needed = requested_amount
            for datafile_and_info in datafiles_and_info:
                datafile_and_info.amount_requested = round(fraction * datafile_and_info.amount_available)
                still_needed -= datafile_and_info.amount_requested
            # correcting for possible rounding mistakes.
            for datafile_and_info in datafiles_and_info:
                if still_needed == 0:
                    break
                if still_needed > 0 and datafile_and_info.amount_available > datafile_and_info.amount_requested:
                    available_and_needed = min(datafile_and_info.amount_available - datafile_and_info.amount_requested, still_needed)
                    datafile_and_info.amount_requested += available_and_needed
                    still_needed -= available_and_needed
                if still_needed < 0 and datafile_and_info.amount_requested > 0:
                    to_remove = min(datafile_and_info.amount_requested, -still_needed)
                    datafile_and_info.amount_requested -= to_remove
                    still_needed += to_remove
                # datafiles_and_info[-1].amount_requested += still_needed
            assert still_needed == 0

            information[split_name].append(dict(
                selector=sample.selector,
                reify=sample.reify,
                remove_qualifiers=sample.remove_qualifiers,
                loaded=[
                    dict(
                        file=file_and_info.source_file.relative_to(data_root).as_posix(),
                        amount=file_and_info.amount_requested,
                        hash=file_and_info.hashcode,
                    )
                    for file_and_info in datafiles_and_info
                ],
            ))

            for datafile_and_info in datafiles_and_info:
                all_samples.append(__OneFileDataset(datafile_and_info.source_file, datafile_and_info.amount_requested, sample.reify, sample.remove_qualifiers))
        if all_samples:
            dataset = torch.utils.data.ConcatDataset(all_samples) if split else __EmptyDataSet()
            datasets[split_name] = dataset
        else:
            logger.warning(f"No samples for split {split_name}")
    return datasets, information


def read_queries_from_proto(
    input_path: Path,
    reify: bool,
    remove_qualifiers: bool,
) -> Iterable[QueryData]:
    """Yield query data from a protobuf."""
    assert not (reify and remove_qualifiers), "cannot both reify and remove qualifiers"
    if reify:
        yield from read_queries_from_proto_with_reification(input_path)
    else:
        yield from read_queries_from_proto_without_reification(input_path, remove_qualifiers)


def read_queries_from_proto_with_reification(
    input_path: Path,
) -> Iterable[QueryData]:
    """
    Read preprocessed queries from Protobuf file.
    Then, all triples are reified and qualifiers are added as properties of the blank nodes.

    :param input_path:
        The input file path.

    :yields:
        A query data object.
    """
    input_path = Path(input_path).expanduser().resolve()
    logger.info(f"Reading from {input_path}")

    qd = pb_QueryData()
    with input_path.open("rb") as input_file:
        qd.ParseFromString(input_file.read())

    logger.error("Reification does currently just take the diameter from the source query without modification")

    for query in qd.queries:
        # we reify by reifying each triple and then adding a triple for each qualifier.
        number_of_original_triples = len(query.triples)
        number_of_triples = number_of_original_triples * 3 + len(query.qualifiers)
        number_of_qualifiers = 0
        b = TensorBuilder(number_of_triples, number_of_qualifiers)

        # Reified statement need 3 triples, so we need to track where we are in the builder. we do not have qualifiers, so we do not need to track where the triples are
        for (index, triple) in enumerate(query.triples):
            statement_entity_index = get_entity_mapper().get_reified_statement_index(index)
            builder_triple_index = 3 * index
            b.set_subject_predicate_object_ID(builder_triple_index, statement_entity_index, get_relation_mapper().reifiedSubject, triple.subject)

            predicate_entity = get_entity_mapper().get_entity_for_predicate(triple.predicate)
            b.set_subject_predicate_object_ID(builder_triple_index + 1, statement_entity_index, get_relation_mapper().reifiedPredicate, predicate_entity)

            b.set_subject_predicate_object_ID(builder_triple_index + 2, statement_entity_index, get_relation_mapper().reifiedObject, triple.object)

        # build qualifiers. These are now just added as triples
        for (builder_triple_index, qualifier) in zip(range(number_of_original_triples * 3, number_of_triples), query.qualifiers):
            statement_entity_index = get_entity_mapper().get_reified_statement_index(qualifier.corresponding_triple)
            b.set_subject_predicate_object_ID(builder_triple_index, statement_entity_index, qualifier.qualifier_relation, qualifier.qualifier_value)

        # set diameter and targets
        b.set_diameter(query.diameter)
        b.set_targets_ID(query.targets)

        # build the object
        yield b.build()


def read_queries_from_proto_without_reification(
    input_path: Path,
    remove_qualifiers: bool,
) -> Iterable[QueryData]:
    """
    Read preprocessed queries from Protobuf file.

    :param input_path:
        The input file path.
    :param remove_qualifiers:
        Whether to remove qualifiers.

    :yields:
        A query data object.
    """
    input_path = Path(input_path).expanduser().resolve()
    logger.info(f"Reading from {input_path}")

    qd = pb_QueryData()
    with input_path.open("rb") as input_file:
        qd.ParseFromString(input_file.read())

    for query in qd.queries:
        number_of_triples = len(query.triples)
        if remove_qualifiers:
            number_of_qualifiers = 0
        else:
            number_of_qualifiers = len(query.qualifiers)
        b = TensorBuilder(number_of_triples, number_of_qualifiers)

        # build triples
        for (index, triple) in enumerate(query.triples):
            b.set_subject_ID(index, triple.subject)
            b.set_predicate_ID(index, triple.predicate)
            b.set_object_ID(index, triple.object)
        if not remove_qualifiers:
            # build qualifier
            for (index, qualifier) in enumerate(query.qualifiers):
                b.set_qualifier_rel_ID(index, qualifier.qualifier_relation, qualifier.corresponding_triple)
                b.set_qualifier_val_ID(index, qualifier.qualifier_value, qualifier.corresponding_triple)

        # set diameter and targets
        b.set_diameter(query.diameter)
        b.set_targets_ID(query.targets)

        # build the object
        yield b.build()


class __EmptyDataSet(Dataset[QueryData]):
    def __getitem__(self, item):
        raise KeyError(item)

    def __len__(self):
        return 0


class __OneFileDataset(Dataset[QueryData]):
    """A query dataset from one file (=one query pattern)."""

    def __init__(self, path: Path, amount: int, reify: bool, remove_qualifiers: bool):
        assert not (reify and remove_qualifiers), "Cannot both remove qualifiers and reify them."
        super().__init__()
        self.data = list(itertools.islice(read_queries_from_proto(path, reify, remove_qualifiers), amount))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, item: int) -> QueryData:
        return self.data[item]


def _unique_with_inverse(*tensors: LongTensor) -> Tuple[LongTensor, Sequence[LongTensor]]:
    # get unique IDs over all these tensors, and the flat inverse
    unique, flat_inverse = torch.cat([t.view(-1) for t in tensors], dim=0).unique(return_inverse=True)

    # decompose inverse into the individual tensors
    inverse_flat_tensors = flat_inverse.split([t.numel() for t in tensors])
    inverse_list = [
        it.view(*t.shape)
        for t, it in zip(
            tensors,
            inverse_flat_tensors,
        )
    ]

    return unique, inverse_list


@dataclasses.dataclass
class QueryGraphBatch:
    """A batch of query graphs."""

    #: The global entity IDs occurring in this batch. Their order corresponds to the batch-local entity ID,
    #: i.e. `local_entity_id = i` corresponds to global entity ID `global_entity_ids[i]`.
    #: shape: (num_unique_batch_entities,)
    entity_ids: LongTensor

    #: The global relation IDs occurring in this batch. Their order corresponds to the batch-local relation ID.
    #: shape: (num_unique_batch_relations,)
    relation_ids: LongTensor

    #: The edge index of the batch graph (in batch-local entity IDs)
    #: shape: (2, num_batch_edges)
    edge_index: LongTensor

    #: The edge types of the batch graph (in batch-local relation IDs)
    #: shape: (num_batch_edges,)
    edge_type: LongTensor

    #: The qualifier index of the batch graph (in batch-local relation/entity/edge IDs)
    #: shape: (3, num_batch_qualifier_pairs)
    qualifier_index: LongTensor

    #: shape: (num_unique_batch_entities,)
    #: The index which maps nodes to a particular query/graph, e.g., [0,0,0,1,1,1] for 6 nodes of two queries
    graph_ids: LongTensor

    #: shape: (batch_size,)
    #: The batched query diameters.
    query_diameter: LongTensor

    #: The targets, in format of pairs (graph_id, entity_id)
    #: shape: (2, num_targets)
    targets: LongTensor

    def __post_init__(self):
        assert self.entity_ids is not None
        assert self.relation_ids is not None
        assert self.edge_type is not None
        assert self.qualifier_index is not None
        assert self.graph_ids is not None
        assert self.query_diameter is not None
        assert self.targets is not None


def collate_query_data(
    batch: Sequence[QueryData],
) -> QueryGraphBatch:
    """
    A collator for query graph batches.

    :param batch:
        The sequence of batch elements.

    :return:
        A 8-tuple of

        1. global_entity_ids: shape: (num_unique_batch_entities,)
            The global entity IDs occurring in this batch. Their order corresponds to the batch-local entity ID,
            i.e. `local_entity_id = i` corresponds to global entity ID `global_entity_ids[i]`.
        2. global_relation_ids: shape: (num_unique_batch_relations,)
            The global relation IDs occurring in this batch. Their order corresponds to the batch-local relation ID.
        3. batch_local_edge_index, shape: (2, num_batch_edges)
            The edge index of the batch graph.
        4. batch_local_edge_type, shape: (num_batch_edges,)
            The edge types of the batch graph.
        5. batch_local_qualifier_index: shape: (3, num_batch_qualifier_pairs)
            The qualifier index of the batch graph.
        6. batch_graph_ids: shape: (num_unique_batch_entities,)
            The index which maps nodes to a particular query/graph, e.g., [0,0,0,1,1,1] for 6 nodes of two queries
        7. query_diameter: shape: (batch_size,)
            The batched query diameters.
        8. targets: shape: (2, num_targets)
            The targets.
    """
    global_entity_ids = []
    global_relation_ids = []
    edge_index = []
    edge_type = []
    qualifier_index = []
    query_diameter = []
    targets = []

    entity_offset = relation_offset = edge_offset = 0
    for i, query_data in enumerate(batch):
        # append query diameter
        query_diameter.append(query_data.query_diameter)

        # convert to local ids
        global_entity_ids_, (local_edge_index, local_qualifier_entities) = _unique_with_inverse(
            query_data.edge_index,
            query_data.qualifier_index[1],
        )

        # target nodes
        target_mask = global_entity_ids_ == get_entity_mapper().get_target_index()
        global_entity_ids_[target_mask] = get_entity_mapper().highest_entity_index + 1

        # variable nodes
        var_mask = (global_entity_ids_ >= get_entity_mapper().variable_start) & (global_entity_ids_ < get_entity_mapper().variable_start + get_entity_mapper().MAX_VARIABLE_COUNT)
        global_entity_ids_[var_mask] = get_entity_mapper().highest_entity_index + 2

        # blank nodes
        reification_mask = global_entity_ids_ >= get_entity_mapper().reification_start
        global_entity_ids_[reification_mask] = get_entity_mapper().highest_entity_index + 3

        global_relation_ids_, (local_edge_type, local_qualifier_relations) = _unique_with_inverse(
            query_data.edge_type,
            query_data.qualifier_index[0],
        )

        # add offsets: entities ...
        local_edge_index = local_edge_index + entity_offset
        local_qualifier_entities = local_qualifier_entities + entity_offset
        # ... relations ...
        local_edge_type = local_edge_type + relation_offset
        local_qualifier_relations = local_qualifier_relations + relation_offset
        # ... and edge ids
        batch_edge_ids = query_data.qualifier_index[2] + edge_offset

        # re-compose qualifier index
        local_qualifier_index = torch.stack(
            [
                local_qualifier_relations,
                local_qualifier_entities,
                batch_edge_ids,
            ],
            dim=0,
        )

        # append
        global_entity_ids.append(global_entity_ids_)
        global_relation_ids.append(global_relation_ids_)
        edge_index.append(local_edge_index)
        edge_type.append(local_edge_type)
        qualifier_index.append(local_qualifier_index)

        # increase counters
        entity_offset += len(global_entity_ids_)
        relation_offset += len(global_relation_ids_)
        edge_offset += len(local_edge_type)

        # collate targets
        targets.append(torch.stack([
            torch.full_like(query_data.targets, fill_value=i),
            query_data.targets,
        ]))

    # concatenate
    global_entity_ids_t = torch.cat(global_entity_ids, dim=-1)
    global_relation_ids_t = torch.cat(global_relation_ids, dim=-1)
    edge_index_t = torch.cat(edge_index, dim=-1)
    edge_type_t = torch.cat(edge_type, dim=-1)
    qualifier_index_t = torch.cat(qualifier_index, dim=-1)
    query_diameter_t = torch.as_tensor(query_diameter, dtype=torch.long)
    targets_t = torch.cat(targets, dim=-1)

    # add graph ids
    graph_ids = torch.empty_like(global_entity_ids_t)
    start = 0
    for i, ids in enumerate(global_entity_ids):
        stop = start + len(ids)
        graph_ids[start:stop] = i
        start = stop

    return QueryGraphBatch(
        entity_ids=global_entity_ids_t,
        relation_ids=global_relation_ids_t,
        edge_index=edge_index_t,
        edge_type=edge_type_t,
        qualifier_index=qualifier_index_t,
        graph_ids=graph_ids,
        query_diameter=query_diameter_t,
        targets=targets_t,
    )


def get_query_data_loaders(
    batch_size: int = 16,
    num_workers: int = 0,
    **kwargs,
) -> Tuple[Mapping[str, torch.utils.data.DataLoader], Mapping[str, Sequence[Mapping]]]:
    """
    Get data loaders for query datasets.

    :param batch_size:
        The batch size to use for all data loaders.
    :param num_workers:
        The number of worker processes to use for loading. 0 means that the data is loaded in the main process.
    :param kwargs:
        Additional keyword-based arguments passed to ``get_query_datasets``.

    :return:
        A pair loaders, information, where loaders is a dictionary from split names to the data loaders, and information
        is a dictionary comprising information about the actually loaded data.
    """
    datasets, information = get_query_datasets(**kwargs)
    loaders = {
        key: DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=key == "train",
            collate_fn=collate_query_data,
            pin_memory=True,
            drop_last=key == "train",
            num_workers=num_workers,
        )
        for key, dataset in datasets.items()
    }
    return loaders, information


def resolve_sample(
    choices: str,
) -> Sample:
    """Resolve a sample from a string representation."""
    split = choices.split(":")
    selector = split[0]
    amount: Union[str, int, Callable[[int], int]]
    if len(split) == 1:
        # there is no amount specification
        raise Exception("No amount specified")
    elif len(split) in {2, 3}:
        amount = split[1]
        if amount == "*":
            pass  # * is a correct psecification
        elif amount.startswith("atmost"):
            amount = amount[6:]
            try:
                # using a different variable beacause of scoping of the nested fucntion.
                maximum_amount = int(amount)

            except ValueError as v:
                raise ValueError("Less than specification can only have an integer as its second part. E.g., *:atmost10000") from v

            def get_at_most(available: int) -> int:
                return min(available, maximum_amount)

            amount = get_at_most
        else:
            # convert to number
            try:
                amount = int(amount)
            except ValueError:
                try:
                    relative_amount = float(amount)
                except ValueError as v:
                    raise ValueError("Could not parse the amount part of the specification " + choices) from v

                def get_relative(absolute: int) -> int:
                    return int(relative_amount * absolute)

                amount = get_relative
    else:
        raise ValueError("sample specification contains more than two ':', specification was " + choices)

    reify = False
    remove_qualifiers = False
    if len(split) == 3:
        # there is an additional option
        option = split[2]
        if option == "reify":
            reify = True
        elif option == "remove_qualifiers":
            remove_qualifiers = True
        else:
            raise Exception(f"Invalid option specified. Expected one of reify|remove_qualifiers, got {option}")

    return Sample(selector, amount, reify=reify, remove_qualifiers=remove_qualifiers)
