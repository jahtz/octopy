# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
from pathlib import Path

import click
from rich.progress import Progress, TextColumn, SpinnerColumn

from ..model import inspect_model


@click.command('inspect')
@click.help_option('--help', hidden=True)
@click.argument(
    'model', 
    type=click.Path(exists=True, dir_okay=False, resolve_path=True, path_type=Path),
    required=True
)
@click.option(
    '-a', '--all', 'output_all',
    help='Print all metadata keys stored in the model file (raw view). '
         'Useful for debugging and for discovering available fields.',
    type=click.BOOL,
    is_flag=True
)
@click.option(
    '-s', '--spec', 'output_spec',
    help='Print the network specification (VGSL) embedded in the model, if present.',
    type=click.BOOL,
    is_flag=True
)
@click.option(
    '-m', '--metrics', 'output_metrics',
    help='Print training metrics stored in the model metadata (e.g. loss/accuracy curves), if present.',
    type=click.BOOL,
    is_flag=True
)
def cli_inspect(model: Path, output_all: bool, output_spec: bool, output_metrics: bool) -> None:
    """
    Inspect a segmentation model file and print selected metadata.
    
    MODEL: Path to the segmentation model file to inspect.
    """
    with Progress(
        SpinnerColumn(),
        TextColumn('[progress.description]{task.description}'),
        transient=True
    ) as progress:
        progress.add_task('Loading', total=None)
        
        metadata = inspect_model(model)
        metadata.pop('accuracy', None)
        if not output_spec and not output_all:
            metadata.pop('vgsl', None)
        if not output_metrics and not output_all:
            metadata.pop('metrics', None)
        print(json.dumps(metadata, indent=2))
