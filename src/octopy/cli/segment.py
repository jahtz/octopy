# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

import click
from pypxml import PageXML

from .util import ClickCallback, spinner, progressbar, read_boolean_environment


logger: logging.Logger = logging.getLogger('octopy')
SHORT_HELP: bool = read_boolean_environment('OCTOPY_VERBOSE_HELP', True)


@click.command('segment')
@click.help_option('--help', hidden=SHORT_HELP)
@click.argument('images', type=click.Path(), callback=ClickCallback.expand_glob, nargs=-1, required=True)
@click.option(
     '-m', '--model',
     help='Path to a custom Kraken segmentation model file. If omitted, Kraken\'s default segmentation model is used.',
     type=click.Path(exists=True, dir_okay=False, resolve_path=True, path_type=Path),
)
@click.option(
    '-o', '--output',
    help='Output directory for generated PAGE-XML files. If omitted, each PAGE-XML file is written next to its input '
         'image.',
    type=click.Path(file_okay=False, path_type=Path)
)
@click.option(
    '-d', '--device',
    help='Compute device for inference (e.g. \'cpu\', \'cuda:0\',...). Use \'auto\' to let Kraken/PyTorch choose.',
    type=click.STRING, 
    default='auto', 
    show_default=True
)
@click.option(
    '-s', '--sort',
    help='Sort regions/lines according to the model\'s reading-order heuristics after segmentation.',
    type=click.BOOL,
    is_flag=True
)
@click.option(
    '--suffix',
    help='Filename suffix (full extension) for generated PAGE-XML files (e.g. \'.xml\' or \'.page.xml\').',
    type=click.STRING, 
    default='.xml', 
    show_default=True
)
@click.option(
    '--mode',
    help='Segmentation output to generate. The effective output is limited by what the selected model provides.',
    type=click.Choice(['lines', 'regions', 'all']), 
    default='all', 
    show_default=True
)
@click.option(
    '--direction',
    help='Principal text direction to assume for the page. This influences reading order and some post-processing.',
    type=click.Choice(['horizontal-lr', 'horizontal-rl', 'vertical-lr', 'vertical-rl']), 
    default='horizontal-lr', 
    show_default=True,
    hidden=SHORT_HELP
)
@click.option(
    '--precision',
    help='Numeric precision for inference. Lower precision can be faster on supported hardware, but may slightly '
         'affect results.',
    type=click.Choice(['transformer-engine', 'transformer-engine-float16', '16-true', '16-mixed', 'bf16-true', 'bf16-mixed', '32-true', '64-true']),
    default='32-true', 
    show_default=True, 
    hidden=SHORT_HELP
)
@click.option(
    '--threads',
    help='Maximum size of the OpenMP/BLAS thread pool used during inference. Increase for throughput on CPU; keep low '
         'to reduce contention.',
    type=click.IntRange(1), 
    default=1, 
    show_default=True, 
    hidden=SHORT_HELP
)
@click.option(
    '--polygonizer', 'polygonizer',
    help='Set the type of polygonizer used for baseline segmentation. \'kraken\' uses the default polygonizer, '
         '\'kraken_fix\' follows the original behavior with minor fixes, and \'octopy\' introduces a completely '
         'redesigned polygonizer.',
    type=click.Choice(['kraken_default', 'kraken_fix', 'octopy']), 
    default='kraken_fix', 
    show_default=True
)
@click.option(
    '--fallback', 'fallback_height',
    help='Fallback bounding box height (in pixels) used when text line polygonization fails. Requires '
         '\'--polygonizer\' to be set to \'kraken_fix\'.',
    type=click.INT,
    default=20
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
    threads: int = 1,
    polygonizer: Literal['kraken_default', 'kraken_fix', 'octopy'] = 'kraken_fix',
    fallback_height: int = 20
) -> None:
    """
    Run Kraken layout analysis (segmentation) on one or more images and write PAGE-XML.
    
    IMAGES: One or more image paths. Glob patterns should be in quotes.
    """
    with spinner as sp:
        sp.add_task('Initialize', total=None)
        from octopy import Segmenter
        segmenter = Segmenter(model, mode, 'octopy', precision, threads, device, polygonizer, fallback_height)

    with progressbar as pb:
        task = pb.add_task('', total=len(images), status='')
        for image in sorted(images):
            pb.update(task, status='/'.join(image.parts[-4:]))
            try:
                logger.info(f'Segment image {image}')
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
