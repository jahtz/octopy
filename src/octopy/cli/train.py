# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

import click
from pypxml import PageXML

from .util import spinner, progressbar, expand_glob


logger: logging.Logger = logging.getLogger(__name__)


@click.command('train')
@click.help_option('--help', hidden=True)
@click.option(
     "-g", "--gt", "ground_truth",
     help="One or more ground truth PAGE-XML files. Use quotes to enclose glob patterns (e.g., \"*.xml\").",
     type=click.Path(),
     callback=expand_glob, 
     required=True,
     multiple=True
)
@click.option(
     "-e", "--eval", "evaluation",
     help="One or more optional evaluation PAGE-XML files. Use quotes to enclose glob patterns (e.g., \"*.xml\").",
     type=click.Path(),
     callback=expand_glob,
     multiple=True
)
def cli_train() -> None:
    """
    Train a custom segmentation model using Kraken.
    """
    pass