"""Test early stopping."""

import torch
from torch import nn

from mphrqe.early_stopping import EarlyStopper


class _DummyModel(nn.Module):
    """A dummy model for testing."""

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(tuple()))


def test_early_stopper():
    """Test early stopper."""
    stopper = EarlyStopper(
        best_model_path=None,
        patience=2,
        larger_is_better=True,
    )
    best_model_path = stopper.best_model_path
    model = _DummyModel()
    for epoch, value in enumerate([
        0.1,
        0.2,
        0.3,  # best epoch
        0.3,
        0.3,  # stop
    ]):
        result = {"validation": {"avg.hits_at_10": value}}
        # modify model
        model.weight.data = torch.as_tensor(epoch, dtype=model.weight.dtype)
        should_stop = stopper.report_result(result=result, model=model, epoch=epoch)
        # check saving of model
        assert best_model_path.is_file()
        if epoch == 4:
            assert should_stop
    # check stopping
    assert stopper.best_metric == 0.3
    assert stopper.best_epoch == 2
