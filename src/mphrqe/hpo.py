"""Hyperparameter optimization with Optuna."""
import dataclasses
import logging
import pathlib
import pprint
from typing import List, Optional, Sequence, Tuple, cast

import optuna
from class_resolver import Hint, Resolver
from optuna import Trial
from optuna.pruners import BasePruner, MedianPruner, NopPruner

from .data.loader import get_query_data_loaders, resolve_sample
from .data.mapping import get_entity_mapper, get_relation_mapper
from .layer.composition import composition_resolver
from .layer.pooling import SumGraphPooling, graph_pooling_resolver
from .layer.util import activation_resolver
from .layer.weighting import message_weighting_resolver
from .models import StarEQueryEmbeddingModel
from .similarity import similarity_resolver
from .tracking import init_tracker
from .training import optimizer_resolver, train_iter
from .utils import get_from_nested_dict

__all__ = [
    "optimize",
]

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class Objective:
    """An objective for optuna."""
    # Data
    data_root: pathlib.Path
    train_data: List[str]
    validation_data: List[str]
    test_data: List[str]

    use_wandb: bool
    wandb_name: Optional[str] = None

    # Model
    num_layers: Optional[int] = None

    # Optimizer
    optimizer: str = "adam"
    lr_range: Tuple[float, float] = (1.0e-04, 1.0e-02)

    # Training
    num_workers: int = 0
    max_num_epoch: int = 1000

    log2_batch_size_range: Tuple[int, int] = (5, 10)  # [32, 1024]

    # Miscellaneous
    log_level: str = "INFO"

    # Evaluation
    metric: Sequence[str] = ("validation", "avg.hits_at_10")

    def __call__(self, trial: Trial) -> float:
        # set log level
        logging.basicConfig(level=self.log_level)

        # Sample configuration
        embedding_dim = trial.suggest_int(name="embedding_dim", low=32, high=256, step=32)
        num_layers = self.num_layers
        if num_layers is None:
            num_layers = trial.suggest_int(name="num_layers", low=2, high=3)
        learning_rate = trial.suggest_float(name="learning_rate", low=self.lr_range[0], high=self.lr_range[1], log=True)
        log2_batch_size = trial.suggest_int(name="log2_batch_size", low=self.log2_batch_size_range[0], high=self.log2_batch_size_range[1])
        batch_size = 2 ** log2_batch_size
        similarity = trial.suggest_categorical(name="similarity", choices=similarity_resolver.lookup_dict.keys())
        if num_layers > 0:
            # trial.suggest_categorical(name="composition", choices=composition_resolver.lookup_dict.keys())
            composition = composition_resolver.normalize("MultiplicationComposition")
            # trial.suggest_categorical(name="qualifier_composition", choices=composition_resolver.lookup_dict.keys())
            qualifier_composition = composition
            message_weighting = trial.suggest_categorical(name="message_weighting", choices=message_weighting_resolver.lookup_dict.keys())
            dropout = trial.suggest_float(name="dropout", low=0.0, high=0.8, step=0.1)
            use_bias = cast(bool, trial.suggest_categorical(name="use_bias", choices=[False, True]))
            activation = trial.suggest_categorical(name="activation", choices=[
                activation_resolver.normalize(name)
                for name in ("LeakyReLU", "Identity", "PReLU", "ReLU")
            ])
            graph_pooling = trial.suggest_categorical(name="graph_pooling", choices=graph_pooling_resolver.lookup_dict.keys())
        else:
            composition = qualifier_composition = message_weighting = activation = None
            use_bias = False
            dropout = 0.0
            # target pooling does not make sense without message passing
            graph_pooling = graph_pooling_resolver.normalize_cls(SumGraphPooling)

        config = dict(
            # tuned parameters
            embedding_dim=embedding_dim,
            num_layers=num_layers,
            learning_rate=learning_rate,
            batch_size=batch_size,
            composition=composition,
            qualifier_composition=qualifier_composition,
            similarity=similarity,
            message_weighting=message_weighting,
            dropout=dropout,
            activation=activation,
            use_bias=use_bias,
            graph_pooling=graph_pooling,
            # fixed parameters
            data_root=self.data_root,
            epochs=self.max_num_epoch,
            optimizer=self.optimizer,
            log_level=self.log_level,
            use_wandb=self.use_wandb,
            train_data=self.train_data,
            validation_data=self.validation_data,
            test_data=self.test_data,
        )
        logger.info(f"Starting run for config:\n{pprint.pformat(config, indent=2)}")

        # We are not using the click resolvers in the annotations because they
        # cannot be serialized to wandb
        optimizer_instance = optimizer_resolver.lookup(self.optimizer)
        composition_instance = composition_resolver.lookup(composition)
        qualifier_composition_instance = composition_resolver.lookup(qualifier_composition)
        similarity_instance = similarity_resolver.lookup(similarity)

        data_loaders, information = get_query_data_loaders(
            data_root=self.data_root,
            train=map(resolve_sample, self.train_data),
            validation=map(resolve_sample, self.validation_data),
            test=map(resolve_sample, []),
            batch_size=batch_size,
            num_workers=self.num_workers,
        )

        for data_loader in data_loaders.values():
            if data_loader:
                assert len(data_loader) >= 1, \
                    f"All data splits used must be larger than the batch size {batch_size}. Could not create a single batch."
        model_instance = StarEQueryEmbeddingModel(
            num_entities=get_entity_mapper().get_largest_embedding_id() + 1,
            num_relations=get_relation_mapper().get_largest_forward_relation_id() + 1,
            embedding_dim=embedding_dim,
            num_layers=num_layers,
            dropout=dropout,
            activation=activation,
            composition=composition_instance,
            qualifier_composition=qualifier_composition_instance,
            message_weighting=message_weighting,
            use_bias=use_bias,
            graph_pooling=graph_pooling,
        )

        # initialize tracker
        result_callback = init_tracker(
            config=config,
            use_wandb=self.use_wandb,
            wandb_name=self.wandb_name,
            information=information,
            is_hpo=True,
        )

        result = float("nan")
        exit_code = None
        try:
            for epoch, epoch_result in enumerate(train_iter(
                model=model_instance,
                data_loaders=data_loaders,
                similarity=similarity_instance,
                optimizer=optimizer_instance,
                optimizer_kwargs=dict(
                    lr=learning_rate,
                ),
                num_epochs=self.max_num_epoch,
                result_callback=result_callback,
                early_stopper_kwargs=dict(
                    key=self.metric,
                    patience=5,
                    relative_delta=0.02,
                    larger_is_better=True,
                    save_best_model=False,
                ),
                delete_best_model_file=True,
            )):
                logger.info(f"Epoch: {epoch:5}/{self.max_num_epoch:5}: {epoch_result}")
                result_ = get_from_nested_dict(epoch_result, key=self.metric, default=None)
                if result_:
                    trial.report(value=result_, step=epoch)
                    result = result_
        except RuntimeError as error:
            logging.fatal(f"ERROR: {error}")
            exit_code = -1
        finally:
            if self.use_wandb:
                import wandb
                wandb.finish(exit_code)
        return result


pruner_resolver = Resolver.from_subclasses(
    base=BasePruner,
    default=NopPruner,
)


def optimize(
    data_root: pathlib.Path,
    train_data: List[str],
    validation_data: List[str],
    test_data: List[str],
    use_wandb: bool,
    num_workers: int,
    num_trials: Optional[int],
    timeout: Optional[float],
    log_level: str,
    wandb_name: Optional[str] = None,
    pruner: Hint[BasePruner] = MedianPruner,
    metric: Sequence[str] = ("validation", "avg.hits_at_10"),
    direction: str = "maximize",
    num_layers: Optional[int] = None,
):
    """
    Optimize hyperparameters with Optuna.

    :param data_root:
        The data root, i.e., the directory containing the binarized queries.
    :param train_data:
        A train data selector, cf. :class:`mphrqe.data.loader.Sample`.
    :param validation_data:
        A validation data selector, cf. :class:`mphrqe.data.loader.Sample`.
    :param test_data:
        A test data selector, cf. :class:`mphrqe.data.loader.Sample`.
    :param use_wandb:
        Whether to use weights and biases for result tracking.
    :param wandb_name:
        The name for the wandb run.
    :param num_workers:
        The number of worker processes to use for data loading. 0 means that the main process also loads the data.
    :param num_trials:
        The maximum number of trials to run.
    :param timeout:
        An timeout in seconds.
    :param log_level:
        The logging level.
    :param pruner:
        A pruner for trials.
    :param metric:
        The metric to optimize.
    :param direction:
        The direction in which to optimize the metric. Either "maximize" or "minimize".
    :param num_layers:
        The number of layers. If given, this will not be tuned.

    :return:
        The best trial.
    """
    # resolver pruner
    pruner = pruner_resolver.make(pruner)

    # Create a new study.
    study = optuna.create_study(
        storage=None,
        sampler=None,
        pruner=pruner,
        direction=direction,
    )

    # setup objective
    objective = Objective(
        data_root=data_root,
        train_data=train_data,
        test_data=test_data,
        validation_data=validation_data,
        log_level=log_level,
        use_wandb=use_wandb,
        wandb_name=wandb_name,
        num_workers=num_workers,
        metric=metric,
        num_layers=num_layers,
    )

    # Invoke optimization of the objective function.
    study.optimize(
        objective,
        n_trials=num_trials,
        timeout=timeout,
        gc_after_trial=True,
    )

    return study.best_trial
