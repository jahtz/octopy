# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from importlib.metadata import version
import logging
from typing import Literal

import click

from .inspect import cli_inspect
from .segment import cli_segment
from .train import cli_train
from .util import setup_logging

logger: logging.Logger = logging.getLogger(__name__)


@click.group(epilog='Developed at Centre for Philology and Digitality (ZPD), University of Würzburg')
@click.help_option('--help')
@click.version_option(version('octopy'), '--version', prog_name='octopy')
@click.pass_context
@click.option(
     '--logging', 'level',
     help='Set logging level.', 
     type=click.Choice(['ERROR', 'WARNING', 'INFO', 'DEBUG']),
     default='ERROR',
     show_default=True
)
def cli(ctx, level: Literal['ERROR', 'WARNING', 'INFO', 'DEBUG'] = 'ERROR', **kwargs) -> None:
    """
    CLI toolkit for layout analysis of historical prints using Kraken
    """
    setup_logging(level)


cli.add_command(cli_inspect)
cli.add_command(cli_segment)
cli.add_command(cli_train)
