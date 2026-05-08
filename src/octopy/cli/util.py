# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import glob
import logging
from os import getenv
from pathlib import Path
from typing import Literal

import click
from rich.logging import RichHandler


def setup_logging(level: Literal['ERROR', 'WARNING', 'INFO', 'DEBUG']) -> None:
    logging.basicConfig(
        level=level,
        format='%(message)s', 
        datefmt='[%X]', 
        handlers=[RichHandler(markup=True, rich_tracebacks=True)]
    )


def read_boolean_environment(name: str, invert: bool = False) -> bool:
    v: str | None = getenv(name)
    if v is None or v.strip().lower() not in {'1', 'true', 't', 'yes', 'y', 'on'}:
        return True if invert else False
    else:
        return False if invert else True


def parse_device(device: str) -> tuple[str, str | list[int]]:
    """
    Parses the input device string to a pytorch accelerator and device string.
    Args:
        device: Encoded device string (see PyTorch documentation).
    Returns:
        Tuple containing accelerator string and device integer/string.
    """
    auto_devices: list[str] = ['auto', 'cpu', 'mps']
    acc_devices: list[str] = ['cuda', 'tpu', 'hpu', 'ipu']
    if device in auto_devices:
        return device, 'auto'
    elif any([device.startswith(x) for x in acc_devices]):
        acc, dv = device.split(':')
        if acc == 'cuda':
            acc = 'gpu'
        return acc, dv if dv == 'auto' else [int(dv)]
    else:
        raise click.BadParameter(f'Invalid device string: {device}')


def expand_glob(ctx: click.Context, param: click.Parameter, patterns: list[str]) -> list[Path]:
    """ Expand glob expressions in path strings """
    paths: list[Path] = []
    for pattern in patterns:
        if glob.has_magic(pattern):
            for match in glob.iglob(pattern, recursive=True):
                path: Path = Path(match)
                if path.is_file():
                    paths.append(path.resolve())
        else:
            path: Path = Path(pattern)
            if path.is_file() and path.exists():
                paths.append(path.resolve())
    return paths


def merge_mapping(ctx, param, value) -> dict[str, str]:
    """ Parse merge mappings """
    if not value:
        return {}
    rules: dict[str, str] = {}
    for rule in value:
        source, target = rule
        if source in rules:
            raise click.BadOptionUsage(param, f'Invalid format: \'{source}\' cannot be declared multiple times as a source')
        rules[source] = target
    return rules
