# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

import click

from .util import ClickCallback, spinner


logger: logging.Logger = logging.getLogger('octopy')


@click.command('train')
@click.help_option('--help', hidden=True)
@click.option(
     '-g', '--gt', 'training_data',
     help='One or more ground truth PAGE-XML files. Use quotes to enclose glob patterns (e.g., \'*.xml\').',
     type=click.Path(), callback=ClickCallback.expand_glob, multiple=True, required=True
)
@click.option(
     '-e', '--eval', 'evaluation_data',
     help='One or more optional evaluation PAGE-XML files. Use quotes to enclose glob patterns (e.g., \'*.xml\').',
     type=click.Path(), callback=ClickCallback.expand_glob, multiple=True
)
@click.option(
     '-t', '--test', 'test_data',
     help='One or more optional evaluation PAGE-XML files. Use quotes to enclose glob patterns (e.g., \'*.xml\').',
     type=click.Path(), callback=ClickCallback.expand_glob, multiple=True
)
@click.option(
     '-p', '--partition', 'partition',
     help='Split ground truth files into training and evaluation sets if no evaluation files are provided. '
          'Default partition is 90% training, 10% evaluation.',
     type=click.FloatRange(min=0.0, max=1.0), default=0.9, show_default=True
)
@click.option(
     '--workers', 'num_workers',
     help='Number of worker processes for CPU-based training.',
     type=click.IntRange(min=1), default=1, show_default=True
)
@click.option(
     '--augment', 'augment',
     help='Switch to enable input image augmentation.',
     type=click.BOOL, is_flag=True
)
@click.option(
     '--data-bach-size', 'data_batch_size',
     help='Number of items to pack into a single sample',
     type=click.IntRange(min=1), default=1, show_default=True
)
@click.option(
     '--line-width', 'line_width',
     help='Line width in the target segmentation map',
     type=click.IntRange(min=1), default=8, show_default=True
)
@click.option(
     '--baseline-position', 'topline',
     help='Indicator of baseline position in dataset.',
     type=click.Choice(['baseline', 'topline', 'centerline']), 
     callback=ClickCallback.baseline, default='baseline', show_default=True
)
@click.option(
     '-mr', '--merge-regions', 'region_merge',
     help='Region merge mapping. One or more mappings of the form \'-mr SOURCE TARGET\', '
          'where \'SOURCE\' is merged into \'TARGET\'.',
     callback=ClickCallback.merge_mapping, multiple=True, nargs=2, 
)
@click.option(
     '-mb', '--merge-lines', 'line_merge',
     help='Baseline merge mapping. One or more mappings of the form \'-mb SOURCE TARGET\', '
          'where \'SOURCE\' is merged into \'TARGET\'.',
     callback=ClickCallback.merge_mapping, multiple=True, nargs=2,
)
def cli_train(**kwargs) -> None:
     '''
     Train a custom segmentation model using Kraken.
     '''
     with spinner as sp:
          sp.add_task('Initialize', total=None)
          from octopy.train import Trainer, training_model_config, training_data_config
          
          kwargs['num_workers'] = 17   
          trainer = Trainer(
               model_config=training_model_config(**kwargs), 
               data_config=training_data_config(**kwargs),
               load=Path('/home/haitz/Seafile/zpd/models/kraken/blla.mlmodel'),
               console=sp
          )
     
     if not click.confirm('Do you want to continue?'):
          return

     trainer.fit()
