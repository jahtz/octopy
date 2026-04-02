# SPDX-License-Identifier: Apache-2.0
import glob
from pathlib import Path

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
    """ Collection of useful click callback methods """
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
