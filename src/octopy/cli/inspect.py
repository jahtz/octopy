# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
import logging
from pathlib import Path

import click

from .util import spinner


@click.command('inspect')
@click.help_option('--help', hidden=True)
@click.argument(
    'model', 
    type=click.Path(exists=True, dir_okay=False, resolve_path=True, path_type=Path),
    required=True
)
@click.option(
    '-a', '--all', 'output_all',
    help='Output all stored keys',
    is_flag=True
)
@click.option(
    '-s', '--spec', 'output_spec',
    help='Output network specifications (vgsl)',
    is_flag=True
)
@click.option(
    '-m', '--metrics', 'output_metrics',
    help='Output training metrics',
    is_flag=True
)
def cli_inspect(model: Path, output_all: bool = False, output_spec: bool = False, output_metrics: bool = False) -> None:
    """
    Inspect segmentation model metadata
    """
    with spinner as sp:
        sp.add_task('Loading', total=None)
        from kraken.lib.vgsl import TorchVGSLModel
        
        try:
            nn: TorchVGSLModel = TorchVGSLModel.load_model(model)
        except Exception as ex:
            logging.error(f'Could not load model: {ex}')
        
        metadata: dict = nn.user_metadata
        metadata.pop('accuracy', None)
        
        if not output_spec and not output_all:
            metadata.pop('vgsl', None)
        if not output_metrics and not output_all:
            metadata.pop('metrics', None)
        print(json.dumps(metadata, indent=2))
