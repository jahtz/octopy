# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from importlib.metadata import version
import logging

import click
from rich.logging import RichHandler

from .inspect import cli_inspect
from .predict import cli_predict
from .train import cli_train


logger: logging.Logger = logging.getLogger(__name__)


def setup_logging(level: int = 0) -> None:
    logging.basicConfig(
        level=max(10, 40 - (10 * level)),
        format='%(message)s', 
        datefmt='[%X]', 
        handlers=[RichHandler(markup=True, rich_tracebacks=False)],
        force=True
    )
    logging.getLogger('pypxml').setLevel(max(30, 40 - (10 * level)))
    logger.info(f'Logging verbosity set to {logging.getLevelName(logger.getEffectiveLevel())}')


@click.group(epilog='Developed at Centre for Philology and Digitality (ZPD), University of Würzburg')
@click.help_option('--help')
@click.version_option(version('octopy'), '--version', prog_name='octopy')
@click.pass_context
@click.option(
     '-v', '--verbose', 'verbosity',
     help='Set the verbosity level. Use -v for WARNING, -vv for INFO, -vvv for DEBUG. [default: ERROR]', 
     count=True
)
def cli(ctx, verbosity: int, *args, **kwargs) -> None:
    """
    CLI toolkit for layout analysis of historical prints using Kraken
    """
    setup_logging(verbosity)


cli.add_command(cli_inspect)
cli.add_command(cli_predict)
cli.add_command(cli_train)
