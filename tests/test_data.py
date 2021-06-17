"""Tests for data loading."""
import random
from typing import Sequence

import pytest
import torch
import torch.utils.data
from torch_geometric.data import Data as DataGeometric

from mphrqe.data import QueryGraphBatch
from mphrqe.data.converter import QueryData
from mphrqe.data.loader import collate_query_data


def _generate_graph(
    max_entity_id: int,
    max_relation_id: int,
    num_statements: int,
    total_num_qualifier_pairs: int,
) -> QueryData:
    """
    Generate random hyper-relational knowledge graphs.

    :param max_entity_id:
        The maximum global entity ID (excl.)
    :param max_relation_id:
        The maximum global relation ID (excl.)
    :param num_statements:
        The number of statements to generate.
    :param total_num_qualifier_pairs:
        The *total* number of qualifier pairs to generate.
    """
    edge_index = torch.randint(max_entity_id, size=(2, num_statements))
    edge_type = torch.randint(max_relation_id, size=(num_statements,))
    qualifier_index = torch.stack(
        [
            torch.randint(max_relation_id, size=(total_num_qualifier_pairs,)),
            torch.randint(max_entity_id, size=(total_num_qualifier_pairs,)),
            torch.randint(num_statements, size=(total_num_qualifier_pairs,)),
        ],
        dim=0,
    )
    # note: This is not consistent with the actual query graph
    query_diameter = torch.as_tensor(data=random.randint(2, 5), dtype=torch.long)
    targets = torch.as_tensor(data=list(set(random.randrange(max_entity_id) for _ in range(random.randrange(2, 6)))))
    return QueryData(
        edge_index=edge_index,
        edge_type=edge_type,
        qualifier_index=qualifier_index,
        targets=targets,
        query_diameter=query_diameter,
        inverses_already_set=False,
    )


def _get_random_graphs(
    max_entity_id: int = 256,
    max_relation_id: int = 128,
    num_graphs: int = 3,
    min_num_statements: int = 16,
    max_num_statements: int = 32,
) -> Sequence[QueryData]:
    # generate random graphs
    return [
        _generate_graph(
            max_entity_id=max_entity_id,
            max_relation_id=max_relation_id,
            num_statements=random.randint(min_num_statements, max_num_statements),
            total_num_qualifier_pairs=random.randint(min_num_statements, 2 * max_num_statements),
        )
        for _ in range(num_graphs)
    ]


@pytest.mark.full_data
def test_collation():
    """Test batch collation for hyper-relational graphs."""
    max_entity_id = 256
    max_relation_id = 128
    batch_size = 3

    # generate random graphs
    graphs = _get_random_graphs(max_entity_id=max_entity_id, max_relation_id=max_relation_id, num_graphs=batch_size)

    # apply collation
    batch = collate_query_data(batch=graphs)

    # check return type and decompose
    assert isinstance(batch, QueryGraphBatch)

    # check shapes
    # shape: (num_batch_entity_ids,)
    assert batch.entity_ids.ndimension() == 1
    # shape: (num_batch_relation_ids,)
    assert batch.relation_ids.ndimension() == 1
    # shape: (2, num_batch_edges,)
    assert batch.edge_index.ndimension() == 2 and batch.edge_index.shape[0] == 2
    # shape: (num_batch_edges,)
    assert batch.edge_type.ndimension() == 1 and batch.edge_index.shape[1] == batch.edge_type.shape[0]
    # shape: (3, num_batch_qualifier_pairs,)
    assert batch.qualifier_index.ndimension() == 2 and batch.qualifier_index.shape[0] == 3
    # shape: (num_batch_entity_ids,)
    assert batch.graph_ids.shape == batch.entity_ids.shape
    # shape: (batch_size,)
    assert batch.query_diameter.shape == (batch_size,)

    # check global id value range
    assert ((batch.entity_ids >= 0) & (batch.entity_ids < max_entity_id)).all()
    assert ((batch.relation_ids >= 0) & (batch.relation_ids < max_relation_id)).all()

    # check edge index value range
    assert (batch.edge_index >= 0).all()
    assert (batch.edge_index < batch.entity_ids.shape[0]).all()

    # check edge type value range
    assert (batch.edge_type >= 0).all()
    assert (batch.edge_type < batch.relation_ids.shape[0]).all()

    # check qualifier index value range
    assert (batch.qualifier_index >= 0).all()
    # num batch edges
    assert (batch.qualifier_index[2] < batch.edge_index.shape[1]).all()
    # num batch relation ids
    assert (batch.qualifier_index[0] < batch.relation_ids.shape[0]).all()
    # num batch entity ids
    assert (batch.qualifier_index[1] < batch.entity_ids.shape[0]).all()

    # check graph id value range
    assert (batch.graph_ids >= 0).all()
    assert (batch.graph_ids < batch_size).all()

    # check query diameter value range
    assert (batch.query_diameter >= 0).all()


def _check_geometric(data_geometric, global_ids: bool = True):
    assert isinstance(data_geometric, DataGeometric)
    assert hasattr(data_geometric, "edge_index")
    edge_index = data_geometric.edge_index
    assert torch.is_tensor(edge_index)
    assert edge_index.shape[0] == 2
    assert edge_index.ndimension() == 2
    assert hasattr(data_geometric, "edge_type")
    edge_type = data_geometric.edge_type
    assert torch.is_tensor(edge_type)
    assert edge_type.ndimension() == 1
    assert edge_type.shape[0] == edge_index.shape[1]
    assert hasattr(data_geometric, "qualifier_index")
    qualifier_index = data_geometric.qualifier_index
    assert torch.is_tensor(qualifier_index)
    assert qualifier_index.ndimension() == 2
    assert qualifier_index.shape[0] == 3
    assert hasattr(data_geometric, "entity_ids")
    assert (data_geometric.entity_ids is None) == global_ids
    assert hasattr(data_geometric, "relation_ids")
    assert (data_geometric.relation_ids is None) == global_ids
    assert hasattr(data_geometric, "graph_ids")
    assert (data_geometric.graph_ids is None) == global_ids
    assert hasattr(data_geometric, "query_diameter")
    assert torch.is_tensor(data_geometric.query_diameter)
