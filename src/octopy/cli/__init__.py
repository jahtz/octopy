# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from importlib.metadata import version
import logging
from typing import Literal

import click
from rich.logging import RichHandler

from .inspect import cli_inspect
from .segment import cli_segment
from .train import cli_train


logging.basicConfig(
    format='%(message)s',
    datefmt='[%X]',
    handlers=[RichHandler(markup=True)]
)
logger: logging.Logger = logging.getLogger('octopy')


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
    logger.setLevel(level)

cli_main.add_command(cli_inspect)
cli_main.add_command(cli_segment)
cli_main.add_command(cli_train)
