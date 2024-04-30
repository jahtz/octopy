from pathlib import Path

import click


@click.command('bls', short_help='Preprocessing and baseline segmentation.')
@click.help_option('--help', '-h')
@click.argument(
    'files',
    type=click.Path(exists=True, dir_okay=True, file_okay=True, resolve_path=True, path_type=Path),
    required=True
)
@click.option(
    '-o', '--output', '_output',
    help='Output directory. Creates directory if it does not exist.  [default: input directory]',
    type=click.Path(exists=False, dir_okay=True, file_okay=False, resolve_path=True, path_type=Path),
    required=False
)
@click.option(
    '-r', '--regex', '_regex',
    help='Regular expression for selecting input files. Only used when input is a directory.',
    type=click.STRING,
    required=False,
    default='*.png',
    show_default=True,
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

    FILES can be a single file or a directory filtered by --regex option.
    """
    print(kwargs)


@click.command('blstrain', short_help='Train baseline segmentation model.')
@click.help_option('--help', '-h')
@click.argument(
    'ground_truth',
    type=click.Path(exists=True, dir_okay=True, file_okay=False, resolve_path=True, path_type=Path),
    required=True
)
@click.argument(
    'images',
    type=click.Path(exists=True, dir_okay=True, file_okay=False, resolve_path=True, path_type=Path),
    required=False
)
@click.option(
    '-gr', '--gtregex', 'gtregex',
    help='Ground truth regex.',
    type=click.STRING,
    default='.xml',
    show_default=True,
    required=False
)
@click.option(
    '-ir', '--imgregex', 'imgregex',
    help='Image regex.',
    type=click.STRING,
    default='.nrm.png',
    show_default=True,
    required=False
)
@click.option(
    '-d', '--device', 'device',
    help='Set device for computation. Some CUDA version recommended.',
    type=click.STRING,
    default='cpu',
    show_default=True,
    required=False
)
@click.option(
    '-m', '--model', 'model',
    help='Set base model for training.',
    type=click.Path(exists=True, dir_okay=False, file_okay=True, resolve_path=True, path_type=Path),
    required=True
)
@click.option(
    '--eval', 'eval',
    help='Set how many percent of ground truth is used as eval set.',
    type=click.INT,
    default=20,
    required=False,
    show_default=True
)
def _blstrain_cli(**kwargs):
    """
    Train baseline segmentation model.

    Accepting PageXML ground truth data from GROUND_TRUTH directory.

    Set IMAGES directory if matching image files are not withing GROUND_TRUTH directory.

    Image filenames should match imageFilename attribute from ground truth xml files (ignoring suffixes).
    """
    print(kwargs)


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
    prog_name='pageseg',
    message='%(prog)s v%(version)s - Developed at Centre for Philology and Digitality (ZPD), University of Würzburg'
)
def cli(**kwargs):
    pass


cli.add_command(_bls_cli)
cli.add_command(_blstrain_cli)
cli.add_command(_recog_cli)
cli.add_command(_recogtrain_cli)


if __name__ == '__main__':
    """
    Main entry point for pagesegment CLI by calling pagesegment.py directly.
    """
    cli()
