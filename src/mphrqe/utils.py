"""Utility methods."""
from typing import Mapping, Optional, Sequence

__all__ = [
    "get_from_nested_dict",
]


def get_from_nested_dict(d: Mapping, key: Sequence[str], default=None) -> Optional[float]:
    """
    Get a value from a nested dictionary.

    :param d:
        The dictionary.
    :param key:
        The sequence of keys.
    :param default:
        The default value to return if the key does not exist.

    :return:
        The value.
    """
    for k in key:
        if k not in d:
            return default
        d = d[k]
    return d  # type: ignore
