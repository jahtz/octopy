from pathlib import Path

import click

from modules.bls import bls_workflow
from modules.blstrain import blstrain_workflow


@click.command('bls', short_help='Preprocessing and baseline segmentation.')
@click.help_option('--help', '-h')
@click.option(
    '-i', '--input', '_input',
    help='Add files for processing. Supports multiple paths and glob expressions (needs to be in "").',
    required=True,
    multiple=True,
)
@click.option(
    '-o', '--output', 'output',
    help='Output directory. Creates directory if it does not exist.  [default: input directory]',
    type=click.Path(exists=False, dir_okay=True, file_okay=False, resolve_path=True, path_type=Path),
    required=True
)
@click.option(
    '-B', '--bin', 'binarize',
    help='Binarize images. Recommended for segmentation.',
    type=click.BOOL,
    required=False,
    is_flag=True
)
@click.option(
    '-b', '--binsuffix', 'binsuffix',
    help='Set output suffix for binarized files.',
    type=click.STRING,
    required=False,
    default='.bin.png',
    show_default=True
)
@click.option(
    '-N', '--nrm', 'normalize',
    help='Normalize images. Recommended for recognition.',
    type=click.BOOL,
    required=False,
    is_flag=True
)
@click.option(
    '-n', '--nrmsuffix', 'nrmsuffix',
    help='Set output suffix for normalized files.',
    type=click.STRING,
    required=False,
    default='.nrm.png',
    show_default=True
)
@click.option(
    '-S', '--seg', 'segment',
    help='Segment images and write results to PageXML files.',
    type=click.BOOL,
    required=False,
    is_flag=True
)
@click.option(
    '-s', '--segsuffix', 'segsuffix',
    help='Set output suffix for PageXML files.',
    type=click.STRING,
    required=False,
    default='.xml',
    show_default=True
)
@click.option(
    '-m', '--model', 'model',
    help='Kraken segmentation model path.',
    type=click.Path(exists=True, dir_okay=False, file_okay=True, resolve_path=True, path_type=Path),
    required=False
)
@click.option(
    '-d', '--device', 'device',
    help='Set device used for segmentation.',
    type=click.STRING,
    required=False,
    default='cpu',
    show_default=True
)
@click.option(
    '--creator', 'creator',
    help='Set creator attribute of PageXML metadata.',
    type=click.STRING,
    required=False,
    default='ZPD Wuerzburg'
)
@click.option(
    '--scale', 'scale',
    help='Recalculate line polygons after segmentation with factor. Increases compute time significantly.',
    type=click.INT,
    required=False
)
@click.option(
    '--threshold', 'threshold',
    help='Set threshold for image binarization in percent.',
    type=click.INT,
    required=False,
    default=50,
    show_default=True
)
def _bls_cli(**kwargs):
    """
    Preprocessing and baseline segmentation.
    """
    bls_workflow(**kwargs)


@click.command('blstrain', short_help='Train baseline segmentation model.')
@click.help_option('--help', '-h')
@click.option(
    '-i', '--input', '_input',
    help='Ground truth files. Supports multiple paths and glob expressions (needs to be in "").',
    required=True,
    multiple=True,
)
@click.option(
    '-o', '--output', 'output_path',
    help='Output directory.',
    type=click.Path(exists=False, dir_okay=True, file_okay=False, resolve_path=True, path_type=Path),
    required=True
)
@click.option(
    '-t', '--train', '_train',
    help='Additional training files. Supports multiple paths and glob expressions (needs to be in "").',
    required=False,
    multiple=True,
)
@click.option(
    '-e', '--eval', '_eval',
    help='Evaluation files. Supports multiple paths and glob expressions (needs to be in "").',
    required=False,
    multiple=True,
)
@click.option(
    '-n', '--name', 'output_name',
    help='Output model name. Results in name_best.mlmodel',
    type=click.STRING,
    required=False,
    default='foo',
    show_default=True
)
@click.option(
    '-p', '--percentage', 'eval_percentage',
    help='Percentage of ground truth data used for evaluation.',
    type=click.INT,
    required=False,
    default=10,
    show_default=True
)
@click.option(
    '-d', '--device', 'device',
    help='Computation device.',
    type=click.STRING,
    required=False,
    default='cpu',
    show_default=True
)
@click.option(
    '--threads', 'threads',
    help='Number of allocated threads.',
    type=click.INT,
    required=False,
    default=1,
    show_default=True
)
@click.option(
    '-m', '--model', 'base_model',
    help='Base model for training.',
    type=click.Path(exists=True, dir_okay=False, file_okay=True, resolve_path=True, path_type=Path),
    required=True,
)
@click.option(
    '--maxepochs', 'max_epochs',
    help='Max epochs.',
    type=click.INT,
    required=False,
    default=100,
    show_default=True
)
@click.option(
    '--minepochs', 'min_epochs',
    help='Min epochs.',
    type=click.INT,
    required=False,
    default=5,
    show_default=True
)
def _blstrain_cli(**kwargs):
    """
    Train baseline segmentation model.

    GT_FILES should be a directory containing PageXML files and matching image files.
    Images should be binary and have a filename matching the imageFilename attribute
    in the PageXML file. (ignoring suffixes)
    """
    blstrain_workflow(**kwargs)


@click.command('recognize', short_help='Transcribe a set of images.')
@click.help_option('--help', '-h')
def _recog_cli(**kwargs):
    """
    Transcribe a set of images.
    """
    print(kwargs)


@click.command('recogtrain', short_help='Train recognition model.')
@click.help_option('--help', '-h')
def _recogtrain_cli(**kwargs):
    """
    Train recognition model.
    """
    print(kwargs)


@click.group()
@click.help_option('--help', '-h')
@click.version_option(
    '2.0',
    '-v', '--version',
    prog_name='Pagesegment',
    message='\n%(prog)s v%(version)s - Developed at Centre for Philology and Digitality (ZPD), University of Würzburg'
)
def cli(**kwargs):
    pass


cli.add_command(_bls_cli)
cli.add_command(_blstrain_cli)
# cli.add_command(_recog_cli)
# cli.add_command(_recogtrain_cli)


if __name__ == '__main__':
    """
    Main entry point for pagesegment CLI by calling pagesegment.py directly.
    """
    cli()
