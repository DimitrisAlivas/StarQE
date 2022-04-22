"""Command line interface."""
import io
import itertools
import json
import logging
import pathlib
import pickle
import pprint
import re
import secrets
import tempfile
import zipfile
import gdown  # type: ignore
from collections import Counter, defaultdict
from operator import itemgetter
from tempfile import TemporaryDirectory
from typing import DefaultDict, List, Optional, Set, Tuple

import click
import numpy
import pandas
import scipy.constants
import seaborn
import torch
from class_resolver import Hint
from matplotlib import pyplot as plt, ticker
from pykeen.utils import invert_mapping, resolve_device
from pykeen.evaluation.rank_based_evaluator import RANK_REALISTIC
from torch import nn

from .data.config import anzograph_init_root, binary_query_root, triples_root
from .data.converter import StrippedSPARQLResultBuilder, convert_all
from .data.loadTriples import load_data
from .data.loader import get_query_data_loaders, resolve_sample
from .data.mapping import (
    create_mapping, get_entity_mapper,
    get_relation_mapper,
)
from .data.query_executor import execute_queries
from .evaluation import MACRO_AVERAGE, MICRO_AVERAGE, evaluate, evaluate_qualifier_impact, expected_mean_rank_from_csv
from .hpo import optimize
from .layer.aggregation import (
    QualifierAggregation,
    qualifier_aggregation_resolver,
)
from .layer.composition import Composition, composition_resolver
from .layer.pooling import GraphPooling, graph_pooling_resolver
from .layer.util import activation_resolver
from .layer.weighting import MessageWeighting, message_weighting_resolver
from .loss import query_embedding_loss_resolver
from .models import StarEQueryEmbeddingModel
from .oracle_metrics import (
    MicroReducer, OracleMeanRank,
    OracleMeanReciprocalRank,
    create_oracle_hits_at_k_class, optimal_answer_set,
    optimal_answer_set_with_extended_input,
)
from .similarity import Similarity, similarity_resolver
from .tracking import init_tracker
from .training import optimizer_resolver, train_iter

__all__ = [
    "main",
]

logger = logging.getLogger(__name__)

# Data options
option_data_root = click.option(
    "-i",
    "--data-root",
    type=pathlib.Path,
    default=binary_query_root,
)
option_train_data = click.option(
    "-tr",
    "--train-data",
    type=str,
    multiple=True,
    default=["/2hop/1qual:1000"],
)
option_validation_data = click.option(
    "-va",
    "--validation-data",
    type=str,
    multiple=True,
    default=[],
)
option_test_data = click.option(
    "-te",
    "--test-data",
    type=str,
    multiple=True,
    default=[],
)
option_num_workers = click.option(
    "-nw",
    "--num-workers",
    type=int,
    default=0,
)

# Wandb options
option_use_wandb = click.option(
    "-w",
    "--use-wandb",
    is_flag=True,
)
option_wandb_name = click.option(
    "-n",
    "--wandb-name",
    default=None,
)
option_wandb_group = click.option(
    "-wg",
    "--wandb-group",
    default=None,
)

# Logging options
option_log_level = click.option(
    "-ll",
    "--log-level",
    type=click.Choice(["DEBUG", "INFO", "WARNING"], case_sensitive=False),
    default="INFO",
)

# StarE encoder options
option_embedding_dim = click.option(
    "-d",
    "--embedding-dim",
    type=int,
    default=128,
)
option_num_layers = click.option(
    "-nl",
    "--num-layers",
    type=int,
    default=2,
)
option_num_layers_optional = click.option(
    "-nl",
    "--num-layers",
    type=int,
    default=None,
)
option_dropout = click.option(
    "-do",
    "--dropout",
    type=float,
    default=0.3,
)
option_activation = activation_resolver.get_option(
    "-a",
    "--activation",
    default=None,
    as_string=True,
)
option_composition = composition_resolver.get_option(
    "-c",
    "--composition",
    default=None,
    as_string=True,
)
option_qualifier_aggregation = qualifier_aggregation_resolver.get_option(
    "-qa",
    "--qualifier-aggregation",
    default=None,
    as_string=True,
)
option_qualifier_composition = composition_resolver.get_option(
    "-qc",
    "--qualifier-composition",
    default=None,
    as_string=True,
)
option_pooling = graph_pooling_resolver.get_option(
    "-gp",
    "--graph-pooling",
    default=None,
    as_string=True,
)
option_use_bias = click.option(
    "-b",
    "--use-bias",
    type=bool,
    default=False,
)
option_message_weighting = message_weighting_resolver.get_option(
    "-mw",
    "--message-weighting",
    default=None,
    as_string=True,
)
option_edge_dropout = click.option(
    "-ed",
    "--edge-dropout",
    type=float,
    default=0.0,
)
option_repeat_layers_until_diameter = click.option(
    "--repeat-layers-until-diameter",
    is_flag=True,
    default=False,
    help="""If set, the layers used in the message passing will be reused until max(batch.diameter) message passing steps have been performed.
             In effect, the weights will be shared.
            Note that the repetition will be each -num-layers, meaning that if the default is used, two layers are created and this pair will be reused.
        """,
)

option_stop_at_diameter = click.option(
    "--stop-at-diameter",
    is_flag=True,
    default=False,
    help="Stop the message propogation as soon as a number of steps have been performed equal to the diameter fo the query",
)
# Decoder options
option_similarity = similarity_resolver.get_option(
    "-s",
    "--similarity",
    default=None,
    as_string=True,
)

# Training + Optimizer options
option_epochs = click.option(
    "-e",
    "--epochs",
    type=int,
    default=1000,
)
option_learning_rate = click.option(
    "-lr",
    "--learning-rate",
    type=float,
    default=0.001,
)
option_batch_size = click.option(
    "-b",
    "--batch-size",
    type=int,
    default=32,
)
option_optimizer = optimizer_resolver.get_option(
    "-o",
    "--optimizer",
    default=None,
    as_string=True,
)
option_save = click.option(
    "-s",
    "--save",
    is_flag=True,
)
option_model_path = click.option(
    "-mp",
    "--model-path",
    type=pathlib.Path,
    default=None,
)

# HPO options
option_num_trials = click.option(
    "-nt",
    "--num-trials",
    type=int,
    default=None,
)
option_timeout = click.option(
    "-to",
    "--timeout",
    type=float,
    default=None,
)


@click.group()
def main():
    """The main entry point."""


@main.command(name="train")
# data options
@option_data_root
@option_train_data
@option_validation_data
@option_test_data
@option_num_workers
# wandb options
@option_use_wandb
@option_wandb_name
@option_wandb_group
# logging options
@option_log_level
# encoder options
@option_embedding_dim
@option_num_layers
@option_dropout
@option_activation
@option_composition
@option_qualifier_aggregation
@option_qualifier_composition
@option_pooling
@option_use_bias
@option_message_weighting
@option_edge_dropout
@option_repeat_layers_until_diameter
@option_stop_at_diameter
# decoder options
@option_similarity
# optimizer + training options
@option_epochs
@option_learning_rate
@option_batch_size
@option_optimizer
# save options
@option_save
@option_model_path
def train_cli(
    # data
    data_root: pathlib.Path,
    train_data: List[str],
    validation_data: List[str],
    test_data: List[str],
    num_workers: int,
    # wandb
    use_wandb: bool,
    wandb_name: str,
    wandb_group: str,
    # logging
    log_level: str,
    # model
    embedding_dim: int,
    num_layers: int,
    dropout: float,
    activation: Hint[nn.Module],
    composition: Hint[Composition],
    qualifier_aggregation: Hint[QualifierAggregation],
    qualifier_composition: Hint[Composition],
    graph_pooling: Hint[GraphPooling],
    use_bias: bool,
    message_weighting: Hint[MessageWeighting],
    edge_dropout: float,
    repeat_layers_until_diameter: bool,
    stop_at_diameter: bool,
    # optimizer + training
    epochs: int,
    learning_rate: float,
    batch_size: int,
    optimizer: str,
    # decoder
    similarity: Hint[Similarity],
    # saving
    save: bool,
    model_path: Optional[pathlib.Path],
):
    """Train a single model for the given configuration."""
    # set log level
    logging.basicConfig(level=log_level)

    logger.info("Start loading data.")
    data_loaders, information = get_query_data_loaders(
        data_root=data_root,
        train=map(resolve_sample, train_data),
        validation=map(resolve_sample, validation_data),
        test=map(resolve_sample, test_data),
        batch_size=batch_size,
        num_workers=num_workers,
    )

    logger.info("Initializing tracker.")
    config = click.get_current_context().params
    result_callback = init_tracker(
        config=config,
        use_wandb=use_wandb,
        wandb_name=wandb_name,
        information=information,
        wandb_group=wandb_group,
        is_hpo=False,
    )

    model = StarEQueryEmbeddingModel(
        num_entities=get_entity_mapper().get_largest_embedding_id() + 1,
        num_relations=get_relation_mapper().get_largest_forward_relation_id() + 1,
        embedding_dim=embedding_dim,
        num_layers=num_layers,
        dropout=dropout,
        activation=activation,
        composition=composition,
        qualifier_aggregation=qualifier_aggregation,
        qualifier_composition=qualifier_composition,
        use_bias=use_bias,
        message_weighting=message_weighting,
        edge_dropout=edge_dropout,
        repeat_layers_until_diameter=repeat_layers_until_diameter,
        stop_at_diameter=stop_at_diameter,
        graph_pooling=graph_pooling,
    )
    logger.info(f"Initialized model:\n{model}.")

    for epoch, epoch_result in enumerate(train_iter(
        model=model,
        data_loaders=data_loaders,
        similarity=similarity,
        optimizer=optimizer,
        optimizer_kwargs=dict(
            lr=learning_rate,
        ),
        num_epochs=epochs,
        result_callback=result_callback,
        early_stopper_kwargs=dict(
            key=("validation", f"{RANK_REALISTIC}.hits_at_10"),
            patience=5,
            relative_delta=0.02,
            larger_is_better=True,
            save_best_model=True,
        ) if validation_data else None,
    )):
        logger.info(f"Epoch: {epoch:5}/{epochs:5}: {epoch_result}")

    if save:
        model_path = model_path or pathlib.Path.home().joinpath(f"{secrets.token_hex()}.pt")
        model_path = model_path.expanduser().resolve()
        model_path.parent.mkdir(exist_ok=True, parents=True)
        torch.save(dict(
            model=model,
            config=config,
            similarity=similarity,
            data=information,  # store to log train data
        ), model_path, pickle_protocol=pickle.HIGHEST_PROTOCOL)
        logger.info(f"Saved model to {model_path.as_uri()}")

    logger.info("Done")


@main.command(name="evaluate")
# data options
@option_data_root
@option_train_data
@option_validation_data
@option_test_data
@option_num_workers
# wandb options
@option_use_wandb
@option_wandb_name
@option_wandb_group
# evaluation options
@click.option(
    "-b",
    "--batch-size",
    type=int,
    default=None,
)
# logging options
@option_log_level
# load options
@option_model_path
@click.option("--evaluate-faithfulness", default=False, is_flag=True)
def evaluate_cli(
    # data options
    data_root: pathlib.Path,
    train_data: List[str],
    validation_data: List[str],
    test_data: List[str],
    num_workers: int,
    # wandb
    use_wandb: bool,
    wandb_name: str,
    wandb_group: str,
    # evaluation
    batch_size: Optional[int],
    # logging
    log_level: str,
    # saving
    model_path: Optional[pathlib.Path],
    evaluate_faithfulness: bool,
):
    """Evaluate a trained model."""
    # set log level
    logging.basicConfig(level=log_level)

    if evaluate_faithfulness:
        assert len(train_data) == 0

    # Resolve device
    device = resolve_device(device=None)
    logger.info(f"Using device: {device}")

    # Load model
    if model_path is None:
        raise ValueError("Model path has to be provided for evaluation.")
    logger.info(f"Loading model from {model_path.as_uri()}")
    data = torch.load(model_path, map_location=device)
    model, config, train_information = [data[k] for k in ("model", "config", "data")]
    logger.info(
        f"Loaded model, trained on \n{pprint.pformat(dict(train_information))}\n"
        f"using configuration \n{pprint.pformat(config)}\n.",
    )
    train_batch_size = config["batch_size"]
    if batch_size is None:
        logger.info(f"No batch size provided. Using the training batch size: {train_batch_size}")
        batch_size = train_batch_size
    elif batch_size > train_batch_size:
        logger.warning(f"Model was trained with batch size {train_batch_size}, but should be evaluated with a larger one: {batch_size}")

    # Load data
    logger.info("Loading evaluation data.")
    data_loaders, information = get_query_data_loaders(
        data_root=data_root,
        train=map(resolve_sample, train_data) if evaluate_faithfulness else [],
        validation=map(resolve_sample, validation_data),
        test=map(resolve_sample, test_data),
        batch_size=batch_size,
        num_workers=num_workers,
    )
    logger.info(f"Evaluating on: \n{pprint.pformat(dict(information))}\n")

    # instantiate decoder
    similarity = similarity_resolver.make(query=data["similarity"])
    logger.info(f"Instantiated similarity {similarity}")

    # instantiate loss
    loss_instance = query_embedding_loss_resolver.make(query=None)
    logger.info(f"Instantiated loss {loss_instance}")

    # Initialize tracking
    result_callback = init_tracker(
        config=config,
        use_wandb=use_wandb,
        wandb_name=wandb_name,
        wandb_group=wandb_group,
        information=information,
        is_hpo=False,
    )

    result = dict()
    for key, data_loader in data_loaders.items():
        if key == "train" and not evaluate_faithfulness:
            continue
        logger.info(f"Evaluating on {key}")
        result[key] = evaluate(
            data_loader=data_loader,
            model=model,
            similarity=similarity,
            loss=loss_instance,
        )
    logger.info(f"Evaluation result: {pprint.pformat(result, indent=2)}")
    if result_callback:
        result_callback(result)


METRICS = {
    f"{RANK_REALISTIC}.hits_at_10": "H@10",
    f"{RANK_REALISTIC}.mean_reciprocal_rank": "MRR",
    f"{RANK_REALISTIC}.adjusted_mean_rank_index": "AMRI",
}


def _buffered_collation(
    input_root: pathlib.Path,
    buffer_path: pathlib.Path,
    force: bool = False,
    minimum_frequency: int = 0,
) -> pandas.DataFrame:
    # prepare buffer path
    buffer_path = buffer_path.expanduser().resolve()
    buffer_path.parent.mkdir(exist_ok=True, parents=True)
    if buffer_path.is_file() and not force:
        df = pandas.read_csv(buffer_path, sep="\t")
        logger.info(f"Read data in shape {df.shape} from buffer: {buffer_path.as_uri()}")
    else:
        data = []
        for path in input_root.rglob(pattern="results.tsv.gz"):
            df = pandas.read_csv(path, sep="\t")

            # long to wide
            df = df.pivot(index="relation_id", columns=["type", "metric"], values="value")
            diff = df["full"] - df["restricted"]

            # comment: num_ranks should not change
            diff["num"] = df["full"][f"{RANK_REALISTIC}.num_ranks"]

            # wide to long
            diff = pandas.melt(diff.reset_index(), id_vars=["relation_id", "num"])

            # add model ID & pattern
            model_id_, pattern = [path.parents[i].name for i in (0, 1)]
            model_id = int(model_id_)
            diff["pattern"] = pattern
            diff["model_id"] = model_id

            # add information
            diff["average"] = MICRO_AVERAGE
            data.append(diff)
            logger.info(f"Read data in shape {diff.shape} from {path.as_uri()}")
        df = pandas.concat(data, ignore_index=True)

        # normalize data type
        df["num"] = df["num"].astype(int)

        df.to_csv(buffer_path, sep="\t", index=False)

    # select relevant metrics
    df = df[df["metric"].isin(METRICS.keys())]

    # drop infrequent
    logger.info(f"Data shape after selecting only relevant metrics: {df.shape}")
    df = df.loc[df["num"] >= minimum_frequency]
    logger.info(f"Data shape after dropping infrequent: {df.shape}")

    return df


@main.group()
def qualifier_impact():
    """Analyze qualifier impact."""


@qualifier_impact.command(name="measure")
# data options
@option_data_root
@option_test_data
@option_num_workers
# evaluation options
@click.option(
    "-a",
    "--average",
    type=click.Choice(choices=[
        MICRO_AVERAGE,
        MACRO_AVERAGE,
    ], case_sensitive=True),
    default=MICRO_AVERAGE,
)
@click.option(
    "-s",
    "--start-relation-id",
    type=int,
    default=None,
    help="restrict evaluated relations to those whose ID is in [start, end).",
)
@click.option(
    "-e",
    "--end-relation-id",
    type=int,
    default=None,
    help="restrict evaluated relations to those whose ID is in [start, end).",
)
# logging options
@option_log_level
# load options
@option_model_path
@click.option(
    "-o",
    "--output-root",
    type=pathlib.Path,
    default=pathlib.Path.cwd(),
)
def analyze_qualifier_impact_cli(
    # data options
    data_root: pathlib.Path,
    test_data: List[str],
    num_workers: int,
    # evaluation
    average: str,
    start_relation_id: Optional[int],
    end_relation_id: Optional[int],
    # logging
    log_level: str,
    # saving
    model_path: Optional[pathlib.Path],
    output_root: pathlib.Path,
):
    """Evaluate the impact of qualifier relations for a trained model."""
    # set log level
    logging.basicConfig(level=log_level)

    if (start_relation_id is None and end_relation_id is not None) or (end_relation_id is None and start_relation_id is not None):
        raise ValueError("Either none of both of start and end relation ID need to be provided.")
    if start_relation_id is None:
        restrict_relations = None
    else:
        assert end_relation_id is not None
        restrict_relations = range(start_relation_id, end_relation_id)

    # Resolve device
    device = resolve_device(device=None)
    logger.info(f"Using device: {device}")

    # Load model
    if model_path is None:
        raise ValueError("Model path has to be provided for evaluation.")
    logger.info(f"Loading model from {model_path.as_uri()}")
    data = torch.load(model_path, map_location=device)
    model, config, train_information = [data[k] for k in ("model", "config", "data")]
    logger.info(
        f"Loaded model, trained on \n{pprint.pformat(dict(train_information))}\n"
        f"using configuration \n{pprint.pformat(config)}\n.",
    )

    # Load data
    logger.info("Loading evaluation data.")
    data_loaders, information = get_query_data_loaders(
        data_root=data_root,
        train=[],
        validation=[],
        test=map(resolve_sample, test_data),
        batch_size=1,
        num_workers=num_workers,
    )
    logger.info(f"Evaluating on: \n{pprint.pformat(dict(information))}\n")
    test_data_loader = data_loaders["test"]

    # instantiate decoder
    similarity = similarity_resolver.make(query=data["similarity"])
    logger.info(f"Instantiated similarity {similarity}")

    df = evaluate_qualifier_impact(
        data_loader=test_data_loader,
        model=model,
        similarity=similarity,
        average=average,
        restrict_relations=restrict_relations,
    ).sort_values(by=["relation_id", "metric", "type"])

    # normalize output path
    output_root = output_root.expanduser().resolve()

    # ensure directory exists
    output_root.mkdir(exist_ok=True, parents=True)

    # save results
    if restrict_relations is None:
        file_name = "results.tsv.gz"
    else:
        file_name = f"results-{start_relation_id}-{end_relation_id}.tsv.gz"
    output_path = output_root.joinpath(file_name)
    df.to_csv(output_path, sep="\t", index=False)
    logger.info(f"Saved result dataframe of size {df.shape} to {output_path.as_uri()} ({output_path.stat().st_size:,} Bytes)")


@qualifier_impact.command(name="heat")
@option_log_level
@click.option("-i", "--input-root", type=pathlib.Path, default=pathlib.Path.cwd())
@click.option("-b", "--buffer-path", type=pathlib.Path, default=pathlib.Path(tempfile.gettempdir(), "buffer.tsv.gz"))
@click.option("-o", "--output-path", type=pathlib.Path, default=pathlib.Path(tempfile.gettempdir(), "qi-heatmap.pdf"))
@click.option("-m", "--minimum-frequency", type=int, default=0)
@click.option("-c", "--color-map", type=str, default="RdBu")
@click.option("--force", is_flag=True)
def qualifier_impact_heatmap(
    log_level: str,
    input_root: pathlib.Path,
    buffer_path: pathlib.Path,
    output_path: pathlib.Path,
    minimum_frequency: int,
    color_map: str,
    force: bool,
):
    """Plot improvement as heatmap."""
    logging.basicConfig(level=log_level)
    df = _buffered_collation(
        input_root=input_root,
        buffer_path=buffer_path,
        force=force,
        minimum_frequency=minimum_frequency,
    )
    id_to_label = invert_mapping(get_relation_mapper().mapping)
    fig, axes = plt.subplots(
        nrows=len(METRICS),
        sharex=True,
        figsize=(16, 9),
    )

    special_relations: Set[Tuple[int, str]] = set()
    ax: plt.Axes
    for (metric, group), ax in zip(df.groupby(by="metric"), axes):
        group = group.pivot_table(
            index="pattern",
            columns="relation_id",
            values="value",
            # fill_value=0.0,  # TODO: This is not nice
        )
        special_relations.update(
            (i, group.columns[i])
            for i in itertools.chain(
                numpy.nanargmax(group.values, axis=1).tolist(),
                numpy.nanargmin(group.values, axis=1).tolist(),
            )
        )
        v = group.values
        # v_max = numpy.nanmax(numpy.abs(v)).item()
        # v_min = -v_max
        v_min, v_max = -1, 1
        im = ax.imshow(v, aspect="auto", vmin=v_min, vmax=v_max, cmap=color_map)
        ax.set_yticks(range(len(group.index)))
        ax.set_yticklabels(group.index)
        ax.set_ylabel(METRICS[metric])
    local_id, global_id = list(zip(*special_relations))
    axes[-1].set_xticks(local_id)
    axes[-1].set_xticklabels(
        [id_to_label[i].rsplit(":", maxsplit=1)[-1] for i in global_id],
        rotation=90,
        fontsize=8,
    )

    axes[-1].set_xlabel("relation")
    fig.tight_layout()
    fig.colorbar(
        im,
        ax=axes,
        label="Improvement by access to qualifier pairs with relation",
    )

    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(output_path)
    logger.info(f"Saved plot to {output_path.as_uri()}")


@qualifier_impact.command(name="scatter")
@option_log_level
@click.option("-i", "--input-root", type=pathlib.Path, default=pathlib.Path.cwd())
@click.option("-b", "--buffer-path", type=pathlib.Path, default=pathlib.Path(tempfile.gettempdir(), "buffer.tsv.gz"))
@click.option("-o", "--output-path", type=pathlib.Path, default=pathlib.Path(tempfile.gettempdir(), "qi-scatter.pdf"))
@click.option("-m", "--minimum-frequency", type=int, default=0)
@click.option("--force", is_flag=True)
def qualifier_impact_scatter(
    log_level: str,
    input_root: pathlib.Path,
    buffer_path: pathlib.Path,
    output_path: pathlib.Path,
    minimum_frequency: int,
    force: bool,
):
    """Plot relation-specific improvement."""
    logging.basicConfig(level=log_level)
    df = _buffered_collation(
        input_root=input_root,
        buffer_path=buffer_path,
        force=force,
        minimum_frequency=minimum_frequency,
    )

    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # metric-specific
    for metric, group in df.groupby(by="metric"):
        metric = METRICS.get(str(metric), metric)
        group = group.rename(columns=dict(value=metric))
        relation_order = group.groupby(
            by="relation_id",
        ).agg(
            {metric: "mean"},
        ).sort_values(by=metric).reset_index()["relation_id"].reset_index()

        group_tmp = pandas.merge(group, relation_order, on="relation_id")
        grid = seaborn.catplot(
            data=group_tmp,
            x="index",
            hue="pattern",
            jitter=0,
            y=metric,
            s=2.0,
            dodge=True,
            # order=...
        )
        grid.ax.grid()
        grid.savefig(output_path.with_stem(stem=f"{output_path.stem}_{metric}"))  # type: ignore


@qualifier_impact.command(name="pattern")
@option_log_level
@click.option("-i", "--input-root", type=pathlib.Path, default=pathlib.Path.cwd())
@click.option("-b", "--buffer-path", type=pathlib.Path, default=pathlib.Path(tempfile.gettempdir(), "buffer.tsv.gz"))
@click.option("-o", "--output-path", type=pathlib.Path, default=pathlib.Path(tempfile.gettempdir(), "qi-pattern.pdf"))
@click.option("-m", "--minimum-frequency", type=int, default=0)
@click.option("--force", is_flag=True)
@click.option("--aggregate", is_flag=True)
def qualifier_impact_pattern(
    log_level: str,
    input_root: pathlib.Path,
    buffer_path: pathlib.Path,
    output_path: pathlib.Path,
    minimum_frequency: int,
    force: bool,
    aggregate: bool,
):
    """Plot improvement grouped by pattern."""
    logging.basicConfig(level=log_level)
    df = _buffered_collation(
        input_root=input_root,
        buffer_path=buffer_path,
        force=force,
        minimum_frequency=minimum_frequency,
    )
    # pre-aggregate
    if aggregate:
        df = df.groupby(
            by=["pattern", "relation_id", "metric"],
        ).agg(dict(value="mean")).reset_index()
    df = df.rename(columns=dict(value="Improvement"))
    df["metric"] = df["metric"].apply(METRICS.get)
    grid = seaborn.catplot(
        data=df,
        x="pattern",
        y="Improvement",
        col="metric",
        kind="box",
        height=4,
        aspect=1 / scipy.constants.golden_ratio,
    )
    grid.set_xticklabels(rotation=90)
    for ax in grid.axes.flat:
        ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))

    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(exist_ok=True, parents=True)
    grid.savefig(output_path)
    logger.info(f"Saved to {output_path.as_uri()}")


@qualifier_impact.command(name="top")
@option_log_level
@click.option("-i", "--input-root", type=pathlib.Path, default=pathlib.Path.cwd())
@click.option("-b", "--buffer-path", type=pathlib.Path, default=pathlib.Path(tempfile.gettempdir(), "buffer.tsv.gz"))
@click.option("-o", "--output-path", type=pathlib.Path, default=pathlib.Path(tempfile.gettempdir(), "qi-top-relations.tex"))
@click.option("-m", "--minimum-frequency", type=int, default=0)
@click.option("-k", "--k", type=int, default=3)
@click.option("--force", is_flag=True)
@click.option("--bold", is_flag=True)
def qualifier_impact_top(
    log_level: str,
    input_root: pathlib.Path,
    buffer_path: pathlib.Path,
    output_path: pathlib.Path,
    minimum_frequency: int,
    force: bool,
    k: int,
    bold: bool,
):
    """Create table for top/bottom qualifier relations."""
    logging.basicConfig(level=log_level)
    df = _buffered_collation(
        input_root=input_root,
        buffer_path=buffer_path,
        force=force,
        minimum_frequency=minimum_frequency,
    )

    # collect best / worst relations per metric-pattern combination
    data = []
    for (metric, pattern), group in df.groupby(by=["metric", "pattern"]):
        group = group.groupby(by=["relation_id", "pattern"]).agg({"value": ["mean", "std"]})
        group = group.reset_index()
        unique_relations = len(group["relation_id"].unique())
        if unique_relations < 2 * k:
            logger.warning(f"Pattern: {pattern} does only contain {unique_relations} unique relations.")
        group = group.sort_values(by=("value", "mean"))
        group = pandas.concat([group.head(n=k), group.tail(n=k)])
        group["metric"] = metric
        data.append(group)
    df = pandas.concat(data, ignore_index=True)

    # normalize metric names
    df["metric"] = df["metric"].apply(METRICS.get)

    # add relation labels
    id_to_label = invert_mapping(get_relation_mapper().mapping)
    df["relation"] = df["relation_id"].apply(id_to_label.get)
    most_common = set(map(itemgetter(0), Counter(df["relation"]).most_common(n=k)))

    def _special_format(value: float) -> str:
        return f"{value:§>5.2f}".replace("§", r"\phantom{0}")

    def _special_format_signed(value: float) -> str:
        base = _special_format(abs(value))
        if value >= 0:
            sign = "+"
            color = "blue"
        else:
            sign = "-"
            color = "red"
        return rf"{{\color{{{color}}}{sign}{base}}}"

    def _link_relations(uri: str) -> str:
        name = uri.rsplit(":", maxsplit=1)[-1]
        return rf"\href{{{uri}}}{{{name}}}"

    data = []
    for metric, group in df.groupby(by="metric"):
        group[metric] = "$" + (100 * group[("value", "mean")]).apply(_special_format_signed) + r" \pm " + (100 * group[("value", "std")]).apply(_special_format) + "$"
        mask = group["relation"].isin(most_common)
        group["relation"] = group["relation"].apply(_link_relations)
        if bold:
            group.loc[mask, "relation"] = group.loc[mask, "relation"].apply(r"\textbf{{{0}}}".format).values
        group = group.sort_values(by=[("pattern", ""), ("value", "mean")])
        group = group.loc[:, ["pattern", "relation", metric]]
        group = group.set_index("pattern").rename(columns={metric: "improvement"})
        group.columns = pandas.MultiIndex.from_tuples([
            (metric, "relation"),
            (metric, "improvement"),
        ])
        data.append(group)
    df = pandas.concat(data, axis=1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with io.StringIO() as buf, pandas.option_context("max_colwidth", 1000):
        df.to_latex(
            buf=buf,
            escape=False,
            sparsify=True,
            multicolumn_format="c",
            column_format="l" + "lr" * len(METRICS),
        )
        buf.seek(0)
        text = []
        prev_pattern = None
        pat = re.compile(r"^\d.*?&")
        for line in buf.readlines():
            match = pat.match(line)
            if match:
                this_pattern = match.group(0)
                if this_pattern == prev_pattern:
                    line = line.replace(this_pattern, "&", 1)
                else:
                    if prev_pattern is not None:
                        text.append(r"\midrule")
                    prev_pattern = this_pattern
                    line = line.replace(
                        this_pattern,
                        rf"\parbox[t]{{2mm}}{{\multirow{{{2 * k}}}{{*}}{{\rotatebox[origin=c]{{90}}{{{this_pattern[:-1].strip()}}}}}}} &",
                    )
            text.append(line)
    output_path.write_text(data="\n".join(text))
    logger.info(f"Written table to {output_path.as_uri()}")


@main.command(name="optimize")
@option_data_root
@option_train_data
@option_validation_data
@option_test_data
@option_use_wandb
@option_wandb_name
@option_num_workers
@option_log_level
@option_num_trials
@option_timeout
@option_num_layers_optional
def optimize_cli(
    # data options
    data_root: pathlib.Path,
    train_data: List[str],
    validation_data: List[str],
    test_data: List[str],
    num_workers: int,
    # wandb options
    use_wandb: bool,
    wandb_name: str,
    # logging options
    log_level: str,
    # optuna options
    num_trials: Optional[int],
    timeout: Optional[float],
    num_layers: Optional[int],
):
    """Optimize hyperparameters using optuna."""
    result = optimize(
        data_root=data_root,
        train_data=train_data,
        validation_data=validation_data,
        test_data=test_data,
        use_wandb=use_wandb,
        wandb_name=wandb_name,
        num_workers=num_workers,
        num_trials=num_trials,
        timeout=timeout,
        log_level=log_level,
        num_layers=num_layers,
    )
    print(f"Best model parameters = {result}")


@main.group()
def oracle():
    "Compute metrics by the oracle link prediction method"
    pass


@oracle.command(name="optimal_answer_set")
def oracle_optimal_answer_set():
    datasets = [
        "data/queries/1hop/1qual-per-triple/test.csv.gz",
        "data/queries/2hop/1qual-per-triple/test.csv.gz",
        "data/queries/3hop/1qual-per-triple/test.csv.gz",
        "data/queries/2i/1qual-per-triple/test.csv.gz",
        "data/queries/3i/1qual-per-triple/test.csv.gz",
        "data/queries/2i-1hop/1qual-per-triple/test.csv.gz",
        "data/queries/1hop-2i/1qual-per-triple/test.csv.gz",
    ]
    hits_at_10_results = []
    mrr_results = []
    for dataset in datasets:
        result = optimal_answer_set(dataset,
                                    metric_classes=[
                                        create_oracle_hits_at_k_class(10),
                                        OracleMeanReciprocalRank,
                                    ],
                                    reducer=MicroReducer()
                                    )
        hits_at_10_results.append(result[0])
        mrr_results.append(result[1])
        print(f"{dataset} hits: {result[0]} mrr: {result[1]}")
    print("Results with micro reducer")
    print(datasets)
    hits_at_10_results_percent = [f"{hits * 100:.2f}" for hits in hits_at_10_results]

    print(f"hits@10: {hits_at_10_results_percent}")
    mrr_results_percent = [f"{mrr * 100:.2f}" for mrr in mrr_results]
    print(f"MRR: {mrr_results_percent}")


@oracle.command(name="optimal_answer_set_extended")
def oracle_optimal_answer_set_extended():
    patterns = ["1hop", "2hop", "3hop", "2i", "3i", "2i-1hop", "1hop-2i"]
    # patterns = ["1hop", "2hop"]

    metrics = [("Hits@1", create_oracle_hits_at_k_class(1)),
               ("Hits@3", create_oracle_hits_at_k_class(3)),
               ("Hits@10", create_oracle_hits_at_k_class(10)),
               ("Hits@100", create_oracle_hits_at_k_class(100)),
               ("MR", OracleMeanRank),
               ("MRR", OracleMeanReciprocalRank),
               ]

    metric_classes = [metric for (_, metric) in metrics]
    metric_names = [metric_name for (metric_name, _) in metrics]
    results = []

    metric_names.append("AMR")
    metric_names.append("AMRI")
    for pattern in patterns:
        training_data = [f"data/queries/{pattern}/1qual-per-triple/train.csv.gz", f"data/queries/{pattern}/1qual-per-triple/validation.csv.gz", f"data/queries/{pattern}/1qual-per-triple/test.csv.gz"]
        test_data = f"data/queries/{pattern}/1qual-per-triple/test.csv.gz"
        result = optimal_answer_set_with_extended_input(training_data=training_data, test_data=test_data, metric_classes=metric_classes, reducer=MicroReducer())
        mr = result[metric_names.index("MR")]
        expected_mean_rank = expected_mean_rank_from_csv(test_data, MICRO_AVERAGE)
        amr = mr / expected_mean_rank
        result.append(amr)
        amri = 1 - ((mr - 1) / (expected_mean_rank - 1))
        result.append(amri)
        results.append(result)
        # print(f"{pattern} hits: {result[0]} mrr: {result[1]}")
    print("Results with micro reducer")

    print(patterns)
    for (index, metric_name) in enumerate(metric_names):
        results_for_pattern = [pattern[index] for pattern in results]
        print(metric_name)
        print("Oracle ", end="")
        for result_for_pattern in results_for_pattern:
            print(fr"&                ${result_for_pattern * 100:.2f} \phantom{{\pm 0.000}}$", end="")
        print(r" \\")


@main.group()
def preprocess():
    """Preprocessing."""
    pass


@preprocess.command(name="skip-and-download-binary", help="Skip preprocessing and download preprocessed data in binary form.")
@click.option("--file-id", type=str, help="ID of the zip file file to fetch", default='1IB99OUnZDS_sBxgk-SEbH7dKIRTlhzk-')  # it can be found when sharing a file
@click.option('--store-path', type=pathlib.Path, default=binary_query_root, help='directory in which to put the queries')
# @click.option('--unzip', is_flag=True, default=True, help='If you include this flag it will try to unzip the file')
def download_binary_from_drive(file_id: str, store_path: pathlib.Path):
    """Fetch binary query data from google drive."""
    assert not store_path.exists() or not any(store_path.iterdir()), "The store-path must either not exist yet or be an empty directory"
    store_path.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory() as tmpdir:
        filename = pathlib.Path(tmpdir) / 'binaryQueries.zip'
        # we do not use the unzipping capability of gdd because we need more control
        gdown.download(id=file_id, output=filename.as_posix(), quiet=False)
        assert filename.exists(), RuntimeError("The file which should have been downloaded does not seem to exist on the file system")
        assert zipfile.is_zipfile(filename), Exception("The downloaded file does not seem to be a zip file")
        with zipfile.ZipFile(filename, 'r') as z:
            zipinfo = z.infolist()
            done = False
            for possible_directory in zipinfo:
                if possible_directory.is_dir() and possible_directory.filename[:-1] == store_path.parts[-1]:
                    # we store in the parent to avoid the extra nested directory
                    z.extractall(store_path.parent)
                    done = True
                    break
            if not done:
                z.extractall(store_path)


@preprocess.command(help="Report number of queries of the data set splits and patterns")
@option_data_root
def dataset_statistics(data_root: pathlib.Path):
    data_root = pathlib.Path(data_root).expanduser().resolve()
    stats: DefaultDict[str, dict] = defaultdict(dict)
    patterns = ["1hop", "2hop", "3hop", "2i", "3i", "2i-1hop", "1hop-2i"]
    splits = ["train", "validation", "test"]
    for pattern in patterns:
        pattern_folder = data_root / pattern
        one_qual_per_triple = pattern_folder / "1qual-per-triple"
        for split in splits:
            stats_file_path = one_qual_per_triple / f"{split}_stats.json"
            with open(stats_file_path) as stats_file:
                patter_one_qual_per_triple_stats = json.load(stats_file)
            amount_in_file = patter_one_qual_per_triple_stats["count"]
            stats[split][pattern] = amount_in_file
    print(stats)

    print("Latex table output:")
    print("")
    print("Patterns ", end="")
    for pattern in patterns:
        print(f"        & {pattern} ", end="")
    print(r"\\")
    for split in splits:
        print(f"{split} ", end="")
        for pattern in patterns:
            print(fr"&                ${stats[split][pattern]:,}$", end="")
        print(r" \\")


# @preprocess.command(name="all")
# def run_all():
#     """Run all preprocessing steps in order with default settings for the paths."""
#     initialize(None, 'anzograph')
#     mapping()
#     sparql(None, None, None)
#     convert(None, None)


@preprocess.command(name="download_wd50k", help="Download the w50k dataset in RDF* format")
@click.option('--store-path', type=pathlib.Path, default=triples_root, help='directory in which to put the queries')
def download_wd50k(store_path: pathlib.Path):
    """Fetch binary query data from google drive."""
    assert not store_path.exists() or not any(store_path.iterdir()), "The store-path must either not exist yet or be an empty directory"
    store_path.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory() as tmpdir:
        filename = pathlib.Path(tmpdir) / 'wd50k.zip'
        # we do not use the unzipping capability of gdd because we need more control
        file_id = "1GijU6TG-ukq5-afuBQpqoUci-6TOPiEZ"
        gdown.download(id=file_id, output=filename.as_posix(), quiet=False)
        assert filename.exists(), RuntimeError("The file which should have been downloaded does not seem to exist on the file system")
        assert zipfile.is_zipfile(filename), Exception("The downloaded file does not seem to be a zip file")
        with zipfile.ZipFile(filename, 'r') as z:
            zipinfo = z.infolist()
            done = False
            for possible_directory in zipinfo:
                if possible_directory.is_dir() and possible_directory.filename[:-1] == store_path.parts[-1]:
                    # we store in the parent to avoid the extra nested directory
                    z.extractall(store_path.parent)
                    done = True
                    break
            if not done:
                z.extractall(store_path)


@preprocess.command()
@click.option("-s", "--source", type=pathlib.Path, default=anzograph_init_root, help="The source directory containing the textual queries.")
@click.option("-g", "--triplestore", default='anzograph', type=click.Choice(['anzograph']),
              help="The triple store in use. Currently only anzograph is supported.")
def initialize(
    source: pathlib.Path,
    triplestore: str,
):
    """Initialize the triple store with the queries"""

    if triplestore == "graphdb":
        raise Exception("The current loading files for graphdb are outdated")

    try:
        load_data(source_directory=source)
    except Exception as e:
        if triplestore == 'anzograph':
            print(f'''AnzoGraph data must be loaded directly trough the file system. Did you download the files  to
                 {source.joinpath("wd50k_train.ttl")}, {source.joinpath("wd50k_test.ttl")}, and {source.joinpath("wd50k_valid.ttl")}''')
        raise e


@preprocess.command()
def mapping():
    """Create the label to ID mapping"""
    create_mapping()


@preprocess.command()
@click.option("-s", "--source", type=pathlib.Path, default=None, help="The source directory containing the csv queries.")
@click.option("-t", "--target", type=pathlib.Path, default=None, help="The target directory to save the textual form of query results.")
@click.option("-e", "--end-point", type=str, default=None, help="The SPARQL endpoint.")
def get_oracle_results(
    source: pathlib.Path,
    target: pathlib.Path,
    end_point: str
):
    """Run the SPARQL queries to obtain the textual query result."""
    builder = StrippedSPARQLResultBuilder(sparql_endpoint=end_point)
    convert_all(
        source_directory=source,
        target_directory=target,
        builder=builder,
        filter=lambda x: "test" in x
    )


@preprocess.command()
@click.option("-s", "--source", type=pathlib.Path, default=None, help="The source directory containing the SPARQL queries.")
@click.option("-t", "--target", type=pathlib.Path, default=None, help="The target directory to save the textual form of query results.")
@click.option("-e", "--end-point", type=str, default=None, help="The SPARQL endpoint.")
def sparql(
    source: pathlib.Path,
    target: pathlib.Path,
    end_point: str,
):
    """Run the SPARQL queries to obtain the textual query result."""
    execute_queries(
        source_directory=source,
        target_directory=target,
        sparql_endpoint=end_point,
    )


@preprocess.command()
@click.option("-s", "--source", type=pathlib.Path, default=None, help="The source directory containing the textual query graphs.")
@click.option("-t", "--target", type=pathlib.Path, default=None, help="The target directory to save the tensorized query graphs.")
def convert(
    source: pathlib.Path,
    target: pathlib.Path,
):
    """Convert the textual query results to index-based."""
    convert_all(
        source_directory=source,
        target_directory=target,
    )
