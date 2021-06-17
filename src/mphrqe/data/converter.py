"""Convert text queries to numeric binary."""
import csv
import dataclasses
import gzip
import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Generic, Iterable, Mapping, Optional, Sequence, Type, TypeVar

import dill
import torch

from .config import binary_query_root, query_root
from .generated.query_pb2 import Qualifier as pb_Qualifier, Query as pb_Query, QueryData as pb_QueryData, Triple as pb_Triple
from .mapping import EntityMapper, get_entity_mapper, get_relation_mapper
from ..typing import LongTensor

__all__ = [
    "convert_all",
    "QueryData",
    "TensorBuilder",
    "ProtobufBuilder",
]

logger = logging.getLogger(__name__)

T = TypeVar('T')


class BinaryFormBuilder(Generic[T], ABC):
    @classmethod
    @abstractmethod
    def __init__(self, numberOfTriples, numberOfQualifiers):
        pass

    @abstractmethod
    def set_subject(self, index: int, entity: str):
        pass

    @abstractmethod
    def set_object(self, index: int, entity: str):
        pass

    @abstractmethod
    def set_predicate(self, index: int, predicate: str):
        pass

    @abstractmethod
    def set_qualifier_rel(self, tripleIndex: int, index: int, predicate: str):
        pass

    @abstractmethod
    def set_qualifier_val(self, tripleIndex: int, index: int, value: str):
        pass

    @abstractmethod
    def set_targets(self, values: Iterable[str]):
        pass

    @abstractmethod
    def set_diameter(self, diameter: int):
        pass

    @abstractmethod
    def build(self) -> T:
        pass

    @staticmethod
    def get_file_extension() -> str:
        """Get the extension to be used for files stored with the store method. Each deriving class must implement its own static method."""
        raise Exception("Each class deriving must implement its own version fo this.s")

    @abstractmethod
    def store(self, collection: Iterable[T], absolute_target_path: Path):
        pass


class ShapeError(RuntimeError):
    """An error raised if the shape does not match."""

    def __init__(self, actual_shape: Sequence[int], expected_shape: Sequence[Optional[int]]):
        """
        Initialize the error.

        :param actual_shape:
            The actual shape.
        :param expected_shape:
            The expected shape.
        """
        super().__init__(f"Actual shape {actual_shape} does not match expected shape {expected_shape}")


def _check_shape(
    tensor: torch.Tensor,
    expected_shape: Sequence[Optional[int]],
) -> bool:
    """Check the shape of a tensor."""
    actual_shape = tensor.shape
    if len(expected_shape) != len(actual_shape):
        raise ShapeError(actual_shape=actual_shape, expected_shape=expected_shape)

    for dim, e_dim in zip(actual_shape, expected_shape):
        if e_dim is not None and dim != e_dim:
            raise ShapeError(actual_shape=actual_shape, expected_shape=expected_shape)

    return True


@dataclasses.dataclass()
class QueryData:
    """This class represents a single query."""

    #: A tensor of edges. shape: (2, num_edges)
    #: The first row contains the heads, the second row the targets.
    edge_index: LongTensor

    # A tensor of edge types (i.e. relation IDs), shape: (num_edges,)
    edge_type: LongTensor

    # A tensor of shape (3, number_of_qualifier_pairs), one column for each qualifier pair.
    # The column contains (in this order) the id of the qualifier relation,
    # the id of the qualifier value, and the index of the corresponding edge
    qualifier_index: LongTensor

    # A tensor with the ids of all answers for this query, shape: (num_answers,)
    targets: LongTensor

    # The longest shortest path between two nodes in this query graph, a scalar tensor.
    query_diameter: LongTensor

    # A flag to indicate that the inverse relations are already included in the tensors.
    # This is to prevent accidentally calling withInverses twice
    inverses_already_set: bool = False

    def __post_init__(self):
        assert _check_shape(tensor=self.edge_index, expected_shape=(2, None))
        assert _check_shape(tensor=self.edge_type, expected_shape=(None,))
        assert _check_shape(tensor=self.qualifier_index, expected_shape=(3, None))
        assert _check_shape(tensor=self.targets, expected_shape=(None,))
        assert _check_shape(tensor=self.query_diameter, expected_shape=tuple())

    def get_number_of_triples(self) -> int:
        """
        Get number of triples in the query

        Returns
        -------
        number_of_triples int
            The number of triples in the query
        """
        return self.edge_index.size()[1]

    def get_number_of_qualifiers(self) -> int:
        """
        Get number of qualifiers in the query

        Returns
        -------
        number_of_qualifiers int
            The number of qualifiers in the query
        """
        return self.qualifier_index.size()[1]

    def with_inverses(self) -> "QueryData":
        """
        Gives you a copy of this QueryData which has inverses added.

        Raises
        ------
        AssertionError
            If this QueryData already contains inverses.
            You can check the inverses_already_set property to check,
            but usually you should know from the programming context.

        Returns
        -------
        QueryData
            A new QueryData object with the inverse edges (and accompanying qualifiers) set
        """
        assert not self.inverses_already_set
        number_of_riples = self.get_number_of_triples()
        number_of_qualifiers = self.get_number_of_qualifiers()
        # Double the space for edges, edge types and qualifier
        new_edge_index = torch.full((2, number_of_riples * 2), -1, dtype=torch.int)
        new_edge_type = torch.full((number_of_riples * 2,), -1, dtype=torch.int)
        new_qualifier_index = torch.full((3, number_of_qualifiers * 2), -1, dtype=torch.int)
        # copy old values
        new_edge_index[:, 0:number_of_riples] = self.edge_index
        new_edge_type[:, 0:number_of_riples] = self.edge_type
        new_qualifier_index[:, 0:number_of_qualifiers] = self.qualifier_index
        # add the inverse values
        new_edge_index[0, number_of_riples:] = self.edge_index[1]
        new_edge_index[1, number_of_riples:] = self.edge_index[0]
        for index, val in enumerate(self.edge_type):
            new_edge_type[number_of_riples + index] = get_relation_mapper().get_inverse_of_index(val)
        # for the qualifiers, we first copy and then update the indices to the corresponding triples
        new_qualifier_index[:, number_of_qualifiers:] = self.qualifier_index
        new_qualifier_index[2, number_of_qualifiers:] += number_of_riples

        new_targets = self.targets
        new_query_diameter = self.query_diameter

        return QueryData(new_edge_index, new_edge_type, new_qualifier_index, new_targets,
                         new_query_diameter, inverses_already_set=True)


class TensorBuilder(BinaryFormBuilder[QueryData]):
    """A builder for binary forms."""

    # This is a singleton used each time there are no qualifiers in the query
    __EMPTY_QUALIFIERS = torch.full((3, 0), -1, dtype=torch.long)

    targets: Optional[LongTensor]

    def __init__(self, number_of_triples: int, number_of_qualifiers: int):
        """
        Initialize the builder.
        """
        super().__init__(number_of_triples, number_of_qualifiers)
        self.number_of_triples = number_of_triples
        self.number_of_qualifiers = number_of_qualifiers
        # We initialize everything to -1. After adding the data there must not be a single -1 left.
        # Store the subject and object,
        self.edge_index = torch.full((2, number_of_triples), -1, dtype=torch.long)
        # Store the type of each edge,
        self.edge_type = torch.full((number_of_triples,), -1, dtype=torch.long)
        # Store all qualifiers. The first row has the qualifier relation, the second on the value and the third one the triple to which is belongs.
        if number_of_qualifiers == 0:
            self.qualifiers = TensorBuilder.__EMPTY_QUALIFIERS
        else:
            self.qualifiers = torch.full((3, number_of_qualifiers), -1, dtype=torch.long)
        # the diameter of the query
        self.diameter = -1
        # The targets of the query. This size is unknown upfront, so we will just create it when set
        self.targets = None
        self.qual_mapping_valid = True

    def set_subject(self, index: int, entity: str):
        entity_index = get_entity_mapper().lookup(entity)
        # set normal edge
        self.set_subject_ID(index, entity_index)

    def set_subject_ID(self, index: int, entity_index: int):
        assert self.edge_index[0, index] == -1
        self.edge_index[0, index] = entity_index

    def set_object(self, index: int, entity: str):
        entity_index = get_entity_mapper().lookup(entity)
        # set normal edge
        self.set_object_ID(index, entity_index)

    def set_object_ID(self, index: int, entity_index: int):
        assert self.edge_index[1, index] == -1
        self.edge_index[1, index] = entity_index

    def set_predicate(self, index: int, predicate: str):
        # set normal edge
        predicate_index = get_relation_mapper().lookup(predicate)
        self.set_predicate_ID(index, predicate_index)

    def set_predicate_ID(self, index: int, predicate_index: int):
        assert self.edge_type[index] == -1
        self.edge_type[index] = predicate_index

    def set_subject_predicate_object_ID(self, triple_index, subject, predicate, object):
        self.set_subject_ID(triple_index, subject)
        self.set_predicate_ID(triple_index, predicate)
        self.set_object_ID(triple_index, object)

    def set_qualifier_rel(self, triple_index: int, qualifier_index: int, predicate: str) -> None:
        # set forward
        predicate_index = get_relation_mapper().lookup(predicate)
        self.qualifiers[0, qualifier_index] = predicate_index
        self.qualifiers[2, qualifier_index] = triple_index

    def set_qualifier_rel_ID(self, qualifier_index, predicateIndex, tripleIndex):
        """
        WARNING : if you use this method, you can no longer use the normal set_qualifier methods because
        this builder no longer has track of the indices
        """
        self.qual_mapping_valid = False
        self.qualifiers[0, qualifier_index] = predicateIndex
        self.qualifiers[2, qualifier_index] = tripleIndex

    def set_qualifier_val(self, triple_index: int, qualifier_index: int, value: str) -> None:
        value_index = get_entity_mapper().lookup(value)
        # set forward
        self.qualifiers[1, qualifier_index] = value_index

    def set_qualifier_val_ID(self, qualifier_index, value_index, tripleIndex):
        """
        .. warning ::
            if you use this method, you can no longer use the normal set_qualifier methods because
            this builder no longer has track of the indices
        """
        self.qual_mapping_valid = False
        self.qualifiers[1, qualifier_index] = value_index

    def set_targets(self, values: Iterable[str]):
        assert self.targets is None, "the list of targets can only be set once. If it is needed to create this incrementally, this implementations can be changed to first collect and only create the tensor in the final build."
        assert len(list(values)) == len(set(values)), f"Values must be a set got {values}"
        mapped = [get_entity_mapper().lookup(value) for value in values]
        self.targets = torch.as_tensor(mapped, dtype=torch.long)

    def set_targets_ID(self, mapped: Iterable[int]):
        assert len(list(mapped)) == len(set(mapped))
        self.targets = torch.as_tensor(data=mapped, dtype=torch.long)

    def set_diameter(self, diameter: int):
        assert self.diameter == -1, "Setting the diameter twice is likely wrong"
        assert diameter <= self.number_of_triples, "the diameter of the query can never be larger than the number of triples"
        self.diameter = diameter

    def build(self) -> QueryData:
        # checkign that everything is filled
        assert (self.edge_index != -1).all()
        assert (self.edge_type != -1).all()
        assert (self.qualifiers != -1).all()
        assert self.diameter != -1
        assert self.targets is not None
        return QueryData(self.edge_index, self.edge_type, self.qualifiers, self.targets, torch.as_tensor(self.diameter))

    @staticmethod
    def get_file_extension() -> str:
        return ".pickle"

    def store(self, collection: Iterable[QueryData], absolute_target_path: Path):
        torch.save(collection, absolute_target_path, pickle_module=dill, pickle_protocol=dill.HIGHEST_PROTOCOL)


class ProtobufBuilder(BinaryFormBuilder[pb_Query]):  # type: ignore

    def __init__(self, numberOfTriples, numberOfQualifiers):
        """
        Create
        """
        super().__init__(numberOfTriples, numberOfQualifiers)
        self.query = pb_Query()
        for _ in range(numberOfTriples):
            self.query.triples.append(pb_Triple())
        self._is_triple_set = [[False, False, False] for _ in range(numberOfTriples)]
        for _ in range(numberOfQualifiers):
            self.query.qualifiers.append(pb_Qualifier())
        self._is_qual_set = [[False, False] for _ in range(numberOfQualifiers)]
        self.query.diameter = 0
        # we need to keep track which qualifier we are putting where.
        # This maps from a tuple (tripleIndex, index), where tripleIndex the index of the triple and index the rank of the qualifier,
        # to the index of the forward edge qaulifier in self.quals. The reverse must be put at the offset numberOfQualifiers
        self.qual_mapping = {}

    def set_subject_predicate_object(self, tripleIndex: int, subject: str, predicate: str, obj: str):
        self.set_subject(tripleIndex, subject)
        self.set_predicate(tripleIndex, predicate)
        self.set_object(tripleIndex, obj)

    def set_subject(self, index: int, entity: str):
        assert not self._is_triple_set[index][0], f"The subject for triple with index {index} has already been set"
        entity_index = get_entity_mapper().lookup(entity)
        # set normal edge
        self.query.triples[index].subject = entity_index
        self._is_triple_set[index][0] = True

    def set_object(self, index: int, entity: str):
        assert not self._is_triple_set[index][2], f"The object for triple with index {index} has already been set"
        entity_index = get_entity_mapper().lookup(entity)
        # set normal edge
        self.query.triples[index].object = entity_index
        self._is_triple_set[index][2] = True

    def set_predicate(self, index: int, predicate: str):
        assert not self._is_triple_set[index][1], f"The predicate for triple with index {index} has already been set"
        # set normal edge
        predicate_index = get_relation_mapper().lookup(predicate)
        self.query.triples[index].predicate = predicate_index
        self._is_triple_set[index][1] = True

    def set_qualifier_rel_val(self, tripleIndex: int, qualifier_index: int, predicate: str, value: str):
        """
        Set the relation `predicate` and value `value` for the qualifier attached to triple with index `tripleIndex` and qualifier index `qualifier_index`
        """
        self.set_qualifier_rel(tripleIndex, qualifier_index, predicate)
        self.set_qualifier_val(tripleIndex, qualifier_index, value)

    def set_qualifier_rel(self, tripleIndex: int, qualifier_index: int, predicate: str):
        assert not self._is_qual_set[qualifier_index][0], f"The relation for qualifier with index {qualifier_index} has already been set"
        # set forward
        predicateIndex = get_relation_mapper().lookup(predicate)
        self.query.qualifiers[qualifier_index].qualifier_relation = predicateIndex
        self.query.qualifiers[qualifier_index].corresponding_triple = tripleIndex
        self._is_qual_set[qualifier_index][0] = True

    def set_qualifier_val(self, tripleIndex: int, qualifier_index: int, value: str):
        assert not self._is_qual_set[qualifier_index][1], f"The value for qualifier with index {qualifier_index} has already been set"
        valueIndex = get_entity_mapper().lookup(value)
        # set forward
        self.query.qualifiers[qualifier_index].qualifier_value = valueIndex
        self._is_qual_set[qualifier_index][1] = True

    def set_targets(self, values: Iterable[str]):
        assert len(self.query.targets) == 0, \
            "the list of targets can only be set once. If it is needed to create this incrementally, this implementations can be changed to first collect and only create the tensor in the final build."
        assert len(list(values)) == len(set(values)), f"Values must be a set, got {values}"
        mapped = [get_entity_mapper().lookup(value) for value in values]
        self.query.targets.extend(mapped)

    def set_diameter(self, diameter: int):
        assert self.query.diameter == 0, "Setting the diameter twice is likely wrong"
        self.query.diameter = diameter

    def build(self) -> pb_Query:  # type: ignore
        # checking that everything is filled
        for parts in self._is_triple_set:
            assert all(parts)
        for qvqr in self._is_qual_set:
            assert all(qvqr)
        assert self.query.diameter != 0
        return self.query

    @staticmethod
    def get_file_extension() -> str:
        return ".proto"

    def store(self, collection: Iterable[pb_Query], absolute_target_path: Path):  # type: ignore
        # torch.save(collection, absolute_target_path, pickle_module=dill, pickle_protocol=pickle.DEFAULT_PROTOCOL)
        query_data = pb_QueryData()
        query_data.queries.extend(collection)
        with open(absolute_target_path, "wb") as output_file:
            output_file.write(query_data.SerializeToString())


def convert_one(absolute_source_path: Path, absolute_target_path: Path, builderClass: Type[BinaryFormBuilder]) -> Mapping[str, str]:
    """Convert a file of textual queries to binary format."""
    converted = []

    with gzip.open(absolute_source_path, mode="rt", newline="") as input_file:
        reader = csv.DictReader(input_file, dialect="unix", quoting=csv.QUOTE_MINIMAL)
        assert reader.fieldnames is not None
        all_headers = [subpart.strip() for header in reader.fieldnames for subpart in header.split("_")]

        # We count the number of subjects in the headers to get the number of triples
        number_of_triples = sum(1 for header in all_headers if header.strip().startswith("s"))

        # We count the number of query relations in the headers to get the number of triples
        number_of_qualifiers = sum(1 for header in all_headers if header.strip().startswith("qr"))

        for row in reader:
            builder = builderClass(number_of_triples, number_of_qualifiers)
            for (part, value) in row.items():
                sub_parts = part.split("_")

                if sub_parts[-1] == "target":
                    # store the targets
                    targets = value.split("|")
                    assert len(set(targets)) == len(targets), f"Targets must be unique, got {targets}"
                    builder.set_targets(targets)
                    # we override the value with "TARGET"
                    value = EntityMapper.get_target_entity_name()
                    # chop off the target
                    sub_parts = sub_parts[:-1]
                elif sub_parts[-1] == "var":
                    # No special action needed except stripping it off, the value of these variables will be enough to do the right thing
                    assert EntityMapper.is_valid_variable_name(value), f"Got {value}"
                    sub_parts = sub_parts[:-1]
                for subpart in sub_parts:
                    subpart = subpart.strip()
                    # we treat each different: subject, predicate, object, qr, qv
                    if subpart.startswith("s"):
                        # it is a subject
                        triple_index = int(subpart[1:])
                        builder.set_subject(triple_index, value)
                    elif subpart.startswith("p"):
                        triple_index = int(subpart[1:])
                        builder.set_predicate(triple_index, value)
                    elif subpart.startswith("o"):
                        triple_index = int(subpart[1:])
                        builder.set_object(triple_index, value)
                    elif subpart.startswith("qr"):
                        indices = subpart[2:].split("i")
                        triple_index = int(indices[0])
                        qualifier_index = int(indices[1])
                        builder.set_qualifier_rel(triple_index, qualifier_index, value)
                    elif subpart.startswith("qv"):
                        indices = subpart[2:].split("i")
                        triple_index = int(indices[0])
                        qualifier_index = int(indices[1])
                        builder.set_qualifier_val(triple_index, qualifier_index, value)
                    elif subpart == "diameter":
                        builder.set_diameter(int(value))
                    else:
                        logger.warning(f"Unknown column with name \"{subpart}\" - ignored")
            converted.append(builder.build())
    # Only storing left
    # Use highest available protocol for better speed / size
    builder.store(converted, absolute_target_path)
    return {"state": "Success"}


def convert_all(
    source_directory: Optional[Path] = None,
    target_directory: Optional[Path] = None,
    builder: Type[BinaryFormBuilder] = ProtobufBuilder,
) -> None:
    """Convert all textual queries to binary format."""
    source_directory = source_directory or query_root
    target_directory = target_directory or binary_query_root
    if not list(source_directory.rglob("*.csv.gz")):
        logger.warning(f"Empty source directory: {source_directory}")
    for query_file_path in source_directory.rglob("*.csv.gz"):
        # take ".csv.gz" of
        name_stem = Path(query_file_path.stem).stem

        # get absolute source path
        relative_source_path = query_file_path.relative_to(source_directory)
        relative_source_directory = relative_source_path.parent
        absolute_source_directory = source_directory.joinpath(relative_source_directory).resolve()

        # compute the destination path
        absolute_target_directory = target_directory.joinpath(relative_source_directory)
        absolute_target_directory.mkdir(parents=True, exist_ok=True)
        absolute_target_path = absolute_target_directory.joinpath(name_stem).with_suffix(builder.get_file_extension()).resolve()

        logger.info(f"{query_file_path.as_uri()} -> {absolute_target_path.as_uri()}")

        # Read stats
        # TODO: Why do we need this?
        source_stats_file_path = absolute_source_directory.joinpath(name_stem + "_stats").with_suffix(".json").resolve()
        if not source_stats_file_path.is_file():
            raise Exception(f"Stats in {source_stats_file_path} not found")
        with source_stats_file_path.open(mode="r") as stats_file:
            source_stats = json.load(stats_file)

        # check the old hash from the status file
        target_stats_file_path = absolute_target_directory.joinpath(name_stem + "_stats").with_suffix(".json").resolve()

        # we only convert if the there is no such file or the hash has changed (the original query has updated)
        if absolute_target_path.is_file():
            if target_stats_file_path.is_file():
                with target_stats_file_path.open(mode="r") as stats_file:
                    target_stats = json.load(stats_file)
                target_hash = target_stats["hash"]
                source_hash = source_stats["hash"]
                if target_hash == source_hash:
                    logger.info(f"Queries already converted for {query_file_path.as_uri()} and hash matches. Not converting again.")
                    continue
                else:
                    # an old version exists but hashes do not match, we remove it, warn the user
                    logger.warning(f"Queries exist for {query_file_path.as_uri()}, but hash does not match. Removing and regenerating!")
            else:
                logger.warning(f"Queries exist for {query_file_path.as_uri()}, but no stats found. Removing and regenerating!")
            absolute_target_path.unlink()

        # perform the query and store the results
        logger.warning(f"Performing conversion from {query_file_path.as_uri()} to {absolute_target_path.as_uri()}")
        try:
            conversion_stats = convert_one(query_file_path, absolute_target_path, builder)
            # Add them to the sourceStats to augment them with new information
            source_stats["conversion"] = conversion_stats
            try:
                with target_stats_file_path.open(mode="w") as stats_file:
                    # The sourceStats are now augmented with new info, so we write that to the target stats
                    json.dump(source_stats, stats_file)
            except Exception:
                # something went wrong writing the stats file, best to remove it and crash.
                logger.error("Failed writing the stats, removing the file to avoid inconsistent state")
                if target_stats_file_path.exists():
                    target_stats_file_path.unlink()
                raise
        except Exception:
            logger.error("Something went wrong executing the query, removing the output file")
            if target_stats_file_path.exists():
                target_stats_file_path.unlink()
            raise
