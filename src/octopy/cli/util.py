# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import glob
from collections import defaultdict
from os import getenv
from pathlib import Path

import click


def read_boolean_environment(name: str, invert: bool = False) -> bool:
    v: str | None = getenv(name)
    if v is None or v.strip().lower() not in {'1', 'true', 't', 'yes', 'y', 'on'}:
        return bool(invert)
    else:
        return not invert


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


def class_merge(ctx, param, value) -> dict[str, list[str]] | None:
    """ Parse merge mappings """
    if not value:
        return None
    
    rules: dict[str, list[str]] = defaultdict(list)
    for source, target in value:
        rules[target].extend(source.split(','))
    return rules
    

def class_valid(ctx, param, value) -> list[str] | None:
    if not value:
        return None
    return value.split(',')
