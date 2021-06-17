"""
Implementation of early stopping.

cf. https://github.com/pykeen/pykeen/blob/26175b58d4266c7dedc86be9d5b876c3a41df09a/src/pykeen/stoppers/early_stopping.py#L57-L124
"""

import logging
import pathlib
import pickle
import tempfile
import uuid
from typing import Any, Mapping, Optional, Sequence, Union

import torch
from torch import nn

from .utils import get_from_nested_dict

__all__ = [
    "is_improvement",
    "EarlyStopper",
]

logger = logging.getLogger(__name__)


def is_improvement(
    best_value: float,
    current_value: float,
    larger_is_better: bool,
    relative_delta: float = 0.0,
) -> bool:
    """
    Decide whether the current value is an improvement over the best value.
    :param best_value:
        The best value so far.
    :param current_value:
        The current value.
    :param larger_is_better:
        Whether a larger value is better.
    :param relative_delta:
        A minimum relative improvement until it is considered as an improvement.
    :return:
        Whether the current value is better.
    """
    if larger_is_better:
        return current_value > (1.0 + relative_delta) * best_value

    # now: smaller is better
    return current_value < (1.0 - relative_delta) * best_value


class EarlyStopper:
    """The early stopping logic."""

    #: The best result so far
    best_metric: float

    #: The epoch at which the best result occurred
    best_epoch: int

    #: The remaining patience
    remaining_patience: Union[int, float]

    def __init__(
        self,
        best_model_path: Union[None, str, pathlib.Path] = None,
        key: Sequence[str] = ("validation", "avg.hits_at_10"),
        patience: Union[int, float] = float("+inf"),
        relative_delta: float = 0.0,
        larger_is_better: bool = True,
        save_best_model: bool = True,
    ):
        """
        Initialize the stopper.
        :param patience:
            The number of reported results with no improvement after which training will be stopped. If None, do never
            stop.
        :param relative_delta:
            The minimum relative improvement necessary to consider it an improved result
        :param larger_is_better:
            Whether a larger value is better, or a smaller.
        """
        self.patience = self.remaining_patience = patience
        self.relative_delta = relative_delta
        self.larger_is_better = larger_is_better
        self.best_epoch = -1
        self.best_metric = float("-inf") if larger_is_better else float("+inf")
        self.key = key
        if best_model_path is None:
            best_model_path = pathlib.Path(tempfile.gettempdir(), f"best-model-{uuid.uuid4()}.pt")
        best_model_path = pathlib.Path(best_model_path)
        if best_model_path.is_file():
            raise FileExistsError(best_model_path)
        self.best_model_path = best_model_path
        logger.info(f"Storing best model to {self.best_model_path.as_uri()}")
        self.save_best_model = save_best_model

    def report_result(
        self,
        result: Mapping[str, Any],
        model: nn.Module,
        epoch: int,
    ) -> bool:
        """
        Report a result at the given epoch.

        :param result:
            The result dictionary.
        :param model:
            The current model. Will be saved in case of improvement.
        :param epoch:
            The epoch.
        :return:
            Whether to stop the training.
        """
        metric = get_from_nested_dict(result, key=self.key, default=None)
        if metric is None:
            logger.warning(f"result for epoch {epoch} did not contain key: {self.key}")
            return False

        # check for improvement
        if self.best_metric is None or is_improvement(
            best_value=self.best_metric,
            current_value=metric,
            larger_is_better=self.larger_is_better,
            relative_delta=self.relative_delta,
        ):
            self.best_epoch = epoch
            self.best_metric = metric
            self.remaining_patience = self.patience
            if self.save_best_model:
                torch.save(model, self.best_model_path, pickle_protocol=pickle.HIGHEST_PROTOCOL)
            logger.info(f"Updated: {self}")
        else:
            self.remaining_patience -= 1

        # Stop if the result did not improve more than delta for patience evaluations
        if self.remaining_patience <= 0:
            logger.info(
                f"Stopping early at epoch {epoch}. The best result {self.best_metric} occurred at "
                f"epoch {self.best_epoch}.",
            )
            return True

        return False

    def load_best_model(self) -> Optional[nn.Module]:
        """Load the best model."""
        if self.save_best_model:
            return torch.load(self.best_model_path)
        return None

    def delete_best_model(self):
        """Delete the best model."""
        if self.best_model_path.exists():
            # not using `missing_ok=True` because we are running this code on pythin 3.7
            self.best_model_path.unlink()

    def __str__(self):
        return (
            f"{self.__class__.__name__}("
            f"best_epoch={self.best_epoch}, "
            f"best_metric={self.best_metric}, "
            f"remaining_patience={self.remaining_patience}, "
            f"best_model_path={self.best_model_path.as_uri()}, "
            f"save_best_model={self.save_best_model}"
            ")"
        )
