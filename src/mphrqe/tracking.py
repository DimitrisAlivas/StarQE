"""Tracking utilities."""
from textwrap import dedent
from typing import Any, Callable, Mapping, Optional, cast

__all__ = [
    "init_tracker",
]


def init_tracker(
    config: Mapping[str, Any],
    use_wandb: bool,
    information: Mapping[str, Any],
    wandb_name: Optional[str] = None,
    wandb_group: Optional[str] = None,
    is_hpo: bool = False,
) -> Optional[Callable[[Mapping[str, Any]], None]]:
    """
    Initialize the results tracker.

    :param config:
        The configuration to log to the tracker.
    :param use_wandb:
        Whether to use wandb.
    :param wandb_name:
        The wandb experiment name.
    :param wandb_group:
        The wandb group name.
    :param information:
        The data information to log.
    :param is_hpo:
        Whether this is an HPO run and should be grouped under the wandb_name.

    :return:
        A result callback.
    """
    result_callback = None
    if use_wandb:
        try:
            import wandb
        except ImportError as e:
            raise RuntimeError(dedent("""
                Could not import wandb. Did you install it? You can do so by

                    pip install .[wandb]

                or directly

                    pip install wandb
            """)) from e
        name = wandb_name if not is_hpo else None
        group = wandb_name if is_hpo else wandb_group
        wandb_run = cast(
            wandb.wandb_sdk.wandb_run.Run,
            wandb.init(project="stare_query", entity="hyperquery", name=name, reinit=True, group=group),
        )
        # All wandb information needs to be collected and then stored as one action on the root of the config object.
        wandb_run.config.update(config)
        wandb_run.config.update(dict(data_loading=information))

        def wandb_log_callback(result: Mapping[str, Any]) -> None:
            """Wrapper around Run.log."""
            wandb_run.log(dict(result))

        result_callback = wandb_log_callback
    return result_callback
