# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

import click
from pypxml import PageXML

from .util import ClickCallback, spinner, progressbar


logger: logging.Logger = logging.getLogger('octopy')
HIDE: bool = True


@click.command('segment')
@click.help_option('--help', hidden=True)
@click.argument('images', type=click.Path(), callback=ClickCallback.expand_glob, nargs=-1, required=True)
@click.option(
     '-m', '--model',
     help='Custom segmentation model. If no model is provided, the default Kraken blla model is used.',
     type=click.Path(exists=True, dir_okay=False, resolve_path=True, path_type=Path),
)
@click.option(
    '-o', '--output',
    help='Custom output directory for created PAGE-XML files. If not set, the parent directory of each input image is used.',
    type=click.Path(file_okay=False, path_type=Path)
)
@click.option(
    '-d', '--device',
    help='Specify the processing device (e.g. \'cpu\', \'cuda:0\',...). Refer to PyTorch documentation for supported devices.',
    type=click.STRING, default='auto', show_default=True
)
@click.option(
    '-s', '--sort',
    help='Sort regions after segmentation using the model specifications.',
    is_flag=True
)
@click.option(
    '--suffix',
    help='Full extension of the created PAGE-XML files.',
    type=click.STRING, default='.xml', show_default=True
)
@click.option(
    '--mode',
    help='Set segmentation mode. Limited by the input model.',
    type=click.Choice(['lines', 'regions', 'all']), default='all', show_default=True
)
@click.option(
    '--direction',
    help='Principal text direction.',
    type=click.Choice(['horizontal-lr', 'horizontal-rl', 'vertical-lr', 'vertical-rl']), 
    default='horizontal-lr', show_default=True
)
@click.option(
    '--precision',
    help='Numerical precision to use for inference.',
    type=click.Choice(['transformer-engine', 'transformer-engine-float16', '16-true', '16-mixed', 'bf16-true', 'bf16-mixed', '32-true', '64-true']),
    default='32-true', show_default=True, hidden=HIDE
)
@click.option(
    '--threads',
    help='Maximum size of OpenMP/BLAS thread pool.',
    type=click.IntRange(1), default=1, show_default=True, hidden=HIDE
)
def cli_segment(
    images: list[Path],
    model: Path | None = None,
    output: Path | None = None,
    device: str = 'auto',
    sort: bool = False,
    suffix: str = '.xml',
    mode: Literal['lines', 'regions', 'all'] = 'all',
    direction: Literal['horizontal-lr', 'horizontal-rl', 'vertical-lr', 'vertical-rl'] = 'horizontal-lr',
    precision: Literal['transformer-engine', 'transformer-engine-float16', '16-true', '16-mixed', 'bf16-true', 'bf16-mixed', '32-true', '64-true'] = '32-true',
    threads: int = 1
) -> None:
    """
    Perform layout analysis using a segmentation model.
    """
    with spinner as sp:
        sp.add_task('Initialize', total=None)
        from octopy.segment import Segmenter
        segmenter = Segmenter(model, mode, 'octopy', precision, threads, device)

    with progressbar as pb:
        task = pb.add_task('', total=len(images), status='')
        for image in images:
            pb.update(task, status='/'.join(image.parts[-4:]))
            try:
                page: PageXML = segmenter.segment(image, sort, direction)
                if output:
                    out: Path = output.joinpath(image.name.split('.')[0] + suffix)
                else:
                    out: Path = image.parent.joinpath(image.name.split('.')[0] + suffix)
                page.save(out)
            except Exception as err:
                logger.error(f'Cloud not segment image {image.as_posix()}: {err}')
            pb.advance(task)
        pb.update(task, status='Done')
