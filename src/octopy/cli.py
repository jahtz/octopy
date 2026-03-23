# SPDX-License-Identifier: Apache-2.0
from importlib.metadata import version
import json
import logging
from pathlib import Path
from typing import Literal

import click
from kraken.lib.vgsl import TorchVGSLModel
from rich.logging import RichHandler

from .util import spinner


logging.basicConfig(
    format='%(message)s',
    datefmt='[%X]',
    handlers=[RichHandler(markup=True)]
)
logger: logging.Logger = logging.getLogger('octopy')


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
def cli_inspect(model: Path, output_all: bool = False) -> None:
    """
    Inspect segmentation model metadata
    """
    with spinner as sp:
        sp.add_task('Loading model metadata', total=None)
        try:
            nn: TorchVGSLModel = TorchVGSLModel.load_model(model)
        except Exception as ex:
            logging.error(f'Could not load model: {ex}')
        print('Spec:\n{')
        for s in nn.spec.split(' '):
            print('  ' + s)
        print('}\n')
        print('Metadata:')
        metadata: dict = nn.user_metadata
        if not output_all:
            metadata.pop('accuracy', None)
            metadata.pop('metrics', None)
        print(json.dumps(metadata, indent=2))


@click.command('train')
@click.help_option('--help', hidden=True)
def cli_train() -> None:
    """
    Train a segmentation model
    """
    pass


@click.command('segment')
@click.help_option('--help', hidden=True)
def cli_segment() -> None:
    """
    Perform layout analysis using a segmentation model
    """
    pass


@click.group(epilog='Developed at Centre for Philology and Digitality (ZPD), University of Würzburg')
@click.help_option('--help')
@click.version_option(version('octopy'), '--version', prog_name='octopy')
@click.pass_context
@click.option(
     '--logging', 'level',
     help='Set logging level.', 
     type=click.Choice(['ERROR', 'WARNING', 'INFO']),
     default='ERROR',
     show_default=True
)
def cli_main(ctx, level: Literal['ERROR', 'WARNING', 'INFO'] = 'ERROR', **kwargs) -> None:
    """
    CLI toolkit for layout analysis of historical prints using Kraken
    """
    logging.getLogger().setLevel(level)
    
cli_main.add_command(cli_train)
cli_main.add_command(cli_segment)
cli_main.add_command(cli_inspect)
