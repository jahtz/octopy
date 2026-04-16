# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import glob
from pathlib import Path
from typing import Literal

import click
from rich.progress import (
    BarColumn,
    MofNCompleteColumn, 
    Progress, 
    SpinnerColumn,
    TextColumn, 
    TimeElapsedColumn,
    TimeRemainingColumn,
)


spinner: Progress = Progress(
    SpinnerColumn(), 
    TextColumn("[progress.description]{task.description}"), 
    transient=True
)
progressbar: Progress = Progress(
    TextColumn("[progress.description]{task.description}"),
    BarColumn(bar_width=30),
    TextColumn('[progress.percentage]{task.percentage:>3.0f}%'),
    MofNCompleteColumn(),
    TimeElapsedColumn(),
    TimeRemainingColumn(),
    TextColumn('• {task.fields[status]}')
)


class ClickCallback:
    @staticmethod
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

    @staticmethod
    def baseline(ctx: click.Context, param: click.Parameter, value: Literal['baseline', 'topline', 'centerline']) -> bool | None:
        if value == 'baseline':
            return False
        elif value == 'topline':
            return True
        else:
            return None
        
    @staticmethod
    def merge_mapping(ctx, param, value) -> dict[str, str]:
        if not value:
            return {}
        rules: dict[str, str] = {}
        for rule in value:
            source, target = rule
            if source in rules:
                raise click.BadOptionUsage(param, f'Invalid format: \'{source}\' cannot be declared multiple times as a source')
            rules[source] = target
        return rules
