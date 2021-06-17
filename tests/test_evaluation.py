"""Tests for evaluation."""
import torch

from mphrqe.evaluation import MACRO_AVERAGE, MICRO_AVERAGE, RankingMetricAggregator, _Ranks, filter_ranks, score_to_rank_multi_target


def _test_score_to_rank_multi_target(average: str):
    """Actually test score_to_rank_multi_target."""
    generator = torch.manual_seed(seed=42)
    batch_size = 2
    num_entities = 7
    num_positives = 5
    scores = torch.rand(batch_size, num_entities, generator=generator)
    targets = torch.stack([
        torch.randint(high=batch_size, size=(num_positives,)),
        torch.randint(high=num_entities, size=(num_positives,)),
    ], dim=0)
    ranks_ = score_to_rank_multi_target(
        scores=scores,
        targets=targets,
        average=average,
    )
    _verify_ranks(ranks_, average, num_entities, num_positives)


def _verify_ranks(ranks_: _Ranks, average: str, num_entities: int, num_positives: int):
    for ranks in (ranks_.pessimistic, ranks_.optimistic, ranks_.realistic, ranks_.expected_rank):
        assert ranks is not None
        assert ranks.shape == (num_positives,)
        assert (ranks >= 1).all()
        assert (ranks <= num_entities).all()
    assert (ranks_.optimistic <= ranks_.pessimistic).all()
    assert (ranks_.weight is None) == (average == MICRO_AVERAGE)


def test_score_to_rank_multi_target_micro():
    """Test score_to_rank_multi_target with micro averaging."""
    _test_score_to_rank_multi_target(average=MICRO_AVERAGE)


def test_score_to_rank_multi_target_macro():
    """Test score_to_rank_multi_target with macro averaging."""
    _test_score_to_rank_multi_target(average=MACRO_AVERAGE)


def test_score_to_rank_infinity():
    """Test score to rank with infinity scores."""
    batch_size = 2
    num_entities = 7
    num_positives = 5
    scores = torch.full(size=(batch_size, num_entities), fill_value=float("inf"))
    targets = torch.stack([
        torch.randint(high=batch_size, size=(num_positives,)),
        torch.randint(high=num_entities, size=(num_positives,)),
    ], dim=0)
    ranks_ = score_to_rank_multi_target(
        scores=scores,
        targets=targets,
        average=MACRO_AVERAGE,
    )
    _verify_ranks(ranks_, average=MACRO_AVERAGE, num_entities=num_entities, num_positives=num_positives)


def test_score_to_rank_multi_target_manual():
    """Test score_to_rank_multi_target on a manual curated examples."""
    targets = torch.as_tensor(data=[[0, 0], [0, 1], [1, 0]]).t()
    scores = torch.as_tensor(data=[
        [1.0, 2.0, 3.0, 4.0],
        [3.0, 2.0, 3.0, 4.0],
    ])

    # Micro
    expected_expected_rank_micro = torch.as_tensor(data=[2.0, 2.0, 2.5])
    micro_ranks_ = score_to_rank_multi_target(
        scores=scores,
        targets=targets,
        average=MICRO_AVERAGE,
    )
    assert torch.allclose(micro_ranks_.expected_rank, expected_expected_rank_micro)
    assert micro_ranks_.weight is None

    # Macro
    expected_expected_rank_macro = torch.as_tensor(data=[2.0, 2.0, 2.5])
    macro_ranks_ = score_to_rank_multi_target(
        scores=scores,
        targets=targets,
        average=MACRO_AVERAGE,
    )
    assert torch.allclose(macro_ranks_.expected_rank, expected_expected_rank_macro)
    expected_weight = torch.as_tensor(data=[0.5, 0.5, 1.0])
    assert torch.allclose(macro_ranks_.weight, expected_weight)


def _test_evaluator(average: str):
    # reproducible testing
    generator = torch.manual_seed(seed=42)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    evaluator = RankingMetricAggregator(average=average)
    batch_size = 2
    num_entities = 5
    num_batches = 3
    nnz = 4
    num_non_empty_queries = 0
    for _ in range(num_batches):
        scores = torch.rand(batch_size, num_entities, device=device, generator=generator)
        targets = torch.stack([
            torch.randint(high=batch_size, size=(nnz,), device=device, generator=generator),
            torch.randint(high=num_entities, size=(nnz,), device=device, generator=generator),
        ], dim=0)
        num_non_empty_queries += targets[0].unique().shape[0]
        evaluator.process_scores_(
            scores=scores,
            targets=targets,
        )
    results = evaluator.finalize()
    if average == MICRO_AVERAGE:
        expected_num_ranks = num_batches * nnz
    elif average == MACRO_AVERAGE:
        expected_num_ranks = num_non_empty_queries
    else:
        raise ValueError(average)
    assert isinstance(results, dict)
    for key, value in results.items():
        assert isinstance(key, str)
        assert isinstance(value, (int, float))
        if "num_ranks" in key:
            assert value == expected_num_ranks
        elif "adjusted_mean_rank_index" in key:
            assert -1 <= value <= 1
        elif "adjusted_mean_rank" in key:
            assert 0 < value < 2
        elif "mean_rank" in key:  # mean_rank, expected_mean_rank
            assert 1 <= value <= num_entities
        else:  # mean_reciprocal_rank, hits_at_k
            assert 0 <= value <= 1, key


def test_evaluator_micro_average():
    """Test evaluator with micro averaging."""
    _test_evaluator(average=MICRO_AVERAGE)


def test_evaluator_macro_average():
    """Test evaluator with macro averaging."""
    _test_evaluator(average=MACRO_AVERAGE)


def test_filter_ranks_manually():
    """Test filter_ranks."""
    # corner case: every rank is one, everything in same batch
    num_entities = 5
    ranks = torch.ones(size=(num_entities,), dtype=torch.long)
    batch_id = torch.zeros_like(ranks)

    filtered_rank = filter_ranks(
        ranks=ranks,
        batch_id=batch_id,
    )
    assert (filtered_rank >= 1).all()
