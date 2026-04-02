# SPDX-License-Identifier: Apache-2.0
from importlib.metadata import version
import json
import logging
from pathlib import Path
from typing import Literal

import click
from rich.logging import RichHandler

from .util import spinner, ClickCallback


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


@click.command('train')
@click.help_option('--help', hidden=True)
def cli_train() -> None:
    """
    Train a segmentation model
    """
    pass


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
    type=click.STRING,
    default='auto',
    show_default=True
)
@click.option(
    '-s', '--sort',
    help='Sort regions after segmentation using the model specifications.',
    is_flag=True
)
@click.option(
    '--suffix',
    help='Full extension of the created PAGE-XML files.',
    type=click.STRING,
    default='.xml',
    show_default=True
)
@click.option(
    '--mode',
    help='Set segmentation mode. Limited by the input model.',
    type=click.Choice(['lines', 'regions', 'all']),
    default='all',
    show_default=True
)
@click.option(
    '--direction',
    help='Principal text direction.',
    type=click.Choice(['horizontal-lr', 'horizontal-rl', 'vertical-lr', 'vertical-rl']),
    default='horizontal-lr',
    show_default=True
)
@click.option(
    '--precision',
    help='Numerical precision to use for inference.',
    type=click.Choice(['transformer-engine', 'transformer-engine-float16', '16-true', '16-mixed', 'bf16-true', 'bf16-mixed', '32-true', '64-true']),
    default='32-true',
    show_default=True
)
@click.option(
    '--threads',
    help='Maximum size of OpenMP/BLAS thread pool.',
    type=click.IntRange(1),
    default=1,
    show_default=True
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
    Perform layout analysis using a segmentation model
    """
    with spinner as sp:
        sp.add_task('Loading', total=None)
        from .segment import segment

    segment(images, model, output, suffix, direction, precision, device, threads, 'octopy', mode, sort)


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
