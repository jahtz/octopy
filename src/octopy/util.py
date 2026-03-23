# SPDX-License-Identifier: Apache-2.0
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