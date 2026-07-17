# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

import click
from rich.progress import Progress, TextColumn, BarColumn, MofNCompleteColumn, TimeElapsedColumn, TimeRemainingColumn

from .util import read_boolean_environment, expand_glob


logger: logging.Logger = logging.getLogger(__name__)
SHORT_HELP: bool = read_boolean_environment('OCTOPY_EXTENDED_HELP', True)


@click.command('predict')
@click.help_option('--help', hidden=SHORT_HELP)
@click.argument('images', type=click.Path(), callback=expand_glob, nargs=-1, required=True)
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
    help='Compute device for inference (e.g. \'cpu\', \'cuda:0\',...).',
    type=click.STRING, 
    default='cpu', 
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
    '--polygonizer',
    help='Set the type of polygonizer used for baseline segmentation. \'kraken\' uses the default polygonizer, '
         '\'octopy\' follows the original behavior with minor fixes and additions.',
    type=click.Choice(['kraken', 'octopy']), 
    default='octopy', 
    show_default=True
)
@click.option(
    '--line-fallback', 'line_fallback_height',
    help='Fallback bounding box height (in pixels) used when text line polygonization fails. Requires '
         '\'--polygonizer\' to be set to \'octopy\'.',
    type=click.INT,
    default=20
)
@click.option(
    '--creator',
    help='PAGE-XML creator tag.',
    type=click.STRING, 
    default='octopy', 
    show_default=True,
    hidden=SHORT_HELP
)
# TODO: AUTOCAST?
def cli_predict(
    images: list[Path],
    model: Path | None,
    output: Path | None,
    device: str,
    sort: bool,
    suffix: str,
    creator: str,
    mode: Literal['lines', 'regions', 'all'],
    direction: Literal['horizontal-lr', 'horizontal-rl', 'vertical-lr', 'vertical-rl'],
    polygonizer: Literal['kraken', 'octopy'],
    line_fallback_height: int
) -> None:
    """
    Run Kraken layout analysis (segmentation) on one or more images and write PAGE-XML.
    
    IMAGES: One or more image paths. Glob patterns should be in quotes.
    """
    with Progress(
        BarColumn(bar_width=30),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        TextColumn('[progress.description]{task.description}'),
    ) as progress:
        load_task = progress.add_task('Initialize', total=None)
        from octopy.predict import Segmenter
        segmenter = Segmenter(model, device, polygonizer, line_fallback_height)
        progress.remove_task(load_task)
        
        task = progress.add_task('Processing images', total=len(images))
        for fp in images:
            progress.update(task, description='/'.join(fp.parts[-4:]))
            
            try:
                res = segmenter.predict(fp, creator, sort, mode, direction)
                out_dir = output or fp.parent
                out_path: Path = out_dir / f'{fp.name.split(".")[0]}{suffix}'
                res.save(out_path)
            except Exception as err:
                logger.error(f'Cloud not segment image {fp.as_posix()}: {err}')
                
            progress.advance(task)
        progress.update(task, status='Done')
