"""Training loops."""
import logging
from typing import Any, Callable, Iterable, Mapping, Optional

import torch.utils.data
import tqdm
from class_resolver import HintOrType, OptionalKwargs
from pykeen.utils import resolve_device
from pykeen.optimizers import optimizer_resolver

from .data import QueryGraphBatch
from .early_stopping import EarlyStopper
from .evaluation import RankingMetricAggregator, evaluate
from .loss import QueryEmbeddingLoss, query_embedding_loss_resolver
from .models import QueryEmbeddingModel
from .similarity import Similarity, similarity_resolver

__all__ = [
    "optimizer_resolver",
    "train_iter",
]

logger = logging.getLogger(__name__)


def _train_epoch(
    data_loader: torch.utils.data.DataLoader[QueryGraphBatch],
    model: QueryEmbeddingModel,
    loss: QueryEmbeddingLoss,
    similarity: Similarity,
    optimizer: torch.optim.Optimizer,
) -> Mapping[str, float]:
    """
    Train one epoch.

    :param data_loader:
        The training data loader.
    :param model:
        The query embedding model instance.
    :param loss:
        The loss instance.
    :param similarity:
        The similarity instance.
    :param optimizer:
        The optimizer instance.

    :return:
        A dictionary of training metrics.
    """
    # Set model into training mode
    model.train()

    epoch_loss = torch.zeros(size=tuple(), device=model.device)
    train_evaluator = RankingMetricAggregator()
    # batch: QueryGraphBatch
    for batch in tqdm.tqdm(data_loader, desc="Training", unit="batch", unit_scale=True):
        # zero grad
        optimizer.zero_grad()
        # embed query
        x_query = model(batch)
        # compute pairwise similarity to all entities, shape: (batch_size, num_entities)
        # TODO: Replace model.x_e by features?
        scores = similarity(x=x_query, y=model.x_e)
        # now compute the loss based on labels
        targets = batch.targets.to(model.device)
        loss_value = loss(scores, targets)
        # backward pass
        loss_value.backward()
        # update parameters
        optimizer.step()
        # accumulate on device
        epoch_loss += loss_value.detach() * scores.shape[0]
        train_evaluator.process_scores_(scores=scores, targets=targets)
    return dict(
        loss=epoch_loss.item() / len(data_loader),
        **train_evaluator.finalize(),
    )


def train_iter(
    model: QueryEmbeddingModel,
    data_loaders: Mapping[str, torch.utils.data.DataLoader[QueryGraphBatch]],
    loss: HintOrType[QueryEmbeddingLoss] = None,
    similarity: HintOrType[Similarity] = None,
    optimizer: HintOrType[torch.optim.Optimizer] = None,
    optimizer_kwargs: OptionalKwargs = None,
    device: Optional[torch.device] = None,
    num_epochs: int = 1,
    evaluation_frequency: int = 1,
    result_callback: Optional[Callable[[Mapping[str, Any]], None]] = None,
    early_stopper_kwargs: OptionalKwargs = None,
    overwrite_with_best_model: bool = True,
    delete_best_model_file: bool = True,
) -> Iterable[Mapping[str, Mapping[str, float]]]:
    """Train the model, yielding epoch results."""
    # resolve device
    device = resolve_device(device=device)
    logger.info(f"Training on device={device}")

    # send model to device
    model = model.to(device=device)

    # instantiate similarity
    similarity = similarity_resolver.make(similarity)
    logger.info(f"Instantiated similarity {similarity}")

    # instantiate loss
    loss_instance = query_embedding_loss_resolver.make(loss)
    logger.info(f"Instantiated loss {loss_instance}")

    # instantiate optimizer
    optimizer_instance: torch.optim.Optimizer = optimizer_resolver.make(
        optimizer,
        pos_kwargs=optimizer_kwargs,
        params=model.parameters(),
    )
    logger.info(f"Instantiated optimizer {optimizer_instance}")

    early_stopper = EarlyStopper(
        **(early_stopper_kwargs or {}),
    )
    logger.info(f"Instantiated early stopper {early_stopper}")

    train_data_loader = data_loaders["train"]
    for epoch in range(num_epochs):
        logger.info(f"Epoch: {epoch}")
        result = dict(
            train=_train_epoch(
                data_loader=train_data_loader,
                model=model,
                loss=loss_instance,
                similarity=similarity,
                optimizer=optimizer_instance,
            ),
        )

        if (epoch + 1) % evaluation_frequency == 0:
            for key, data_loader in data_loaders.items():
                if key == "train":
                    continue
                logger.info(f"Evaluating on {key}")
                result[key] = evaluate(
                    data_loader=data_loader,
                    model=model,
                    similarity=similarity,
                    loss=loss_instance,
                )
            should_stop = early_stopper.report_result(
                result=result,
                model=model,
                epoch=epoch,
            )
            if should_stop:
                logger.info(f"Stopped training after epoch {epoch} due to early stopper.")
                break

        if result_callback is not None:
            result_callback(result)

        yield result

    # Reset model to best model
    if overwrite_with_best_model:
        best_model = early_stopper.load_best_model()
        if best_model is None:
            logger.warning("Cannot restore weights from best model if the early stopper does not save it!")
        else:
            model.load_state_dict(state_dict=best_model.state_dict())  # type: ignore
    if delete_best_model_file:
        early_stopper.delete_best_model()
