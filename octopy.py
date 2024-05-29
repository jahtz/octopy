import click
from kraken.lib.default_specs import SEGMENTATION_HYPER_PARAMS

from modules.util import validate_merging, parse_files, parse_file
from modules.seg import segment
from modules.segtrain import segtrain
from modules.pp import preprocess


@click.command('seg', short_help='Segment images using Kraken and save the results as XML files.')
@click.help_option('--help', '-h')
@click.argument(
    'files',
    type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True, resolve_path=True),
    callback=parse_files,
    nargs=-1,
)
@click.option(
    '-m', '--model',
    help='Path to segmentation model.',
    type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True, resolve_path=True),
    required=True,
    callback=parse_file
)
@click.option(
    '-o', '--output',
    help='Output directory to save epochs and trained model.',
    type=click.Path(exists=False, file_okay=False, dir_okay=True, resolve_path=True),
    required=True,
    callback=parse_file
)
@click.option(
    '-s', '--suffix', 'output_suffix',
    help='Suffix to append to the output file name. e.g. `.seg.xml` results in `imagename.seg.xml`.',
    type=click.STRING,
    required=False,
    default='.xml',
    show_default=True,
)
@click.option(
    '-d', '--device',
    help='Device to run the model on. (see Kraken guide)',
    type=click.STRING,
    required=False,
    default='cpu',
    show_default=True,
)
@click.option(
    '-c', '--creator',
    help='Creator of the PageXML file.',
    type=click.STRING,
    required=False,
    default='octopy',
)
@click.option(
    '-r', '--recalculate',
    help='Recalculate line polygons with this factor. Increases compute time significantly.',
    type=click.INT,
    required=False,
)
def seg_cli(**kwargs):
    """
    Segment images using Kraken and save the results as XML files.

    Multiple FILES can either be passed by absolute paths or by using wildcards.
    """
    segment(**kwargs)


@click.command('segtrain', short_help='Train a segmentation model using Kraken.')
@click.help_option('--help', '-h')
@click.argument(
    'ground_truth',
    type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True, resolve_path=True),
    callback=parse_files,
    nargs=-1,
)
@click.option(
    '-o', '--output',
    help='Output directory to save epochs and trained model.',
    type=click.Path(exists=False, file_okay=False, dir_okay=True, resolve_path=True),
    required=True,
    callback=parse_file
)
@click.option(
    '-n', '--name', 'model_name',
    help='Name of the output model.',
    type=click.STRING,
    required=False,
    default='foo',
    show_default=True,
)
@click.option(
    '-m', '--model',
    help='Path to existing model to continue training. If set to None, a new model is trained from scratch.',
    type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True, resolve_path=True),
    required=False,
    show_default=False,
)
@click.option(
    '-p', '--partition',
    help='Ground truth data partition ratio between train/validation set.',
    type=click.FLOAT,
    required=False,
    default=0.9,
    show_default=True,
)
@click.option(
    '-d', '--device',
    help='Device to run the model on. (see Kraken guide)',
    type=click.STRING,
    required=False,
    default='cpu',
    show_default=True,
)
@click.option(
    '-t', '--threads',
    help='Number of threads to use (cpu only)',
    type=click.INT,
    required=False,
    default=1,
    show_default=True,
)
@click.option(
    '--max-epochs',
    help='Maximum number of epochs to train.',
    type=click.INT,
    required=False,
    default=SEGMENTATION_HYPER_PARAMS['epochs'],
    show_default=True,
)
@click.option(
    '--min-epochs',
    help='Minimum number of epochs to train.',
    type=click.INT,
    required=False,
    default=SEGMENTATION_HYPER_PARAMS['min_epochs'],
    show_default=True,
)
@click.option(
    '-q',
    '--quit',
    show_default=True,
    default='fixed',
    type=click.Choice(['early', 'fixed']),
    help='Stop condition for training. Set to `early` for early stopping or `fixed` for fixed number of epochs.',
)
@click.option(
    '-v', '--verbose', 'verbosity',
    help='Verbosity level. For level 2 use -vv (0-2)',
    count=True
)
@click.option(
    '-mr', '--merge-regions',
    show_default=True,
    default=None,
    help='Region merge mapping. One or more mappings of the form `$target:$src` where $src is merged into $target.',
    multiple=True,
    callback=validate_merging
)
def segtrain_cli(**kwargs):
    """
    Train a segmentation model using Kraken.

    Multiple GROUND_TRUTH files can either be passed by absolute paths or by using wildcards.
    """
    segtrain(**kwargs)


@click.command('pp', short_help='Preprocess images using Kraken and Ocropus.')
@click.help_option('--help', '-h')
@click.argument(
    'files',
    type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True, resolve_path=True),
    callback=parse_files,
    nargs=-1,
)
@click.option(
    '-o', '--output',
    help='Output directory to save the pre-processed files.',
    type=click.Path(exists=False, file_okay=False, dir_okay=True, resolve_path=True),
    required=True,
    callback=parse_file
)
@click.option(
    '-b', '--binarize', 'bin',
    help='Binarize images.',
    is_flag=True,
    required=False,
)
@click.option(
    '-n', '--normalize', 'nrm',
    help='Normalize images.',
    is_flag=True,
    required=False,
)
@click.option(
    '-r', '--resize', 'res',
    help='Resize images. Used for binarization and normalization.',
    is_flag=True,
    required=False,
)
@click.option(
    '--height',
    help='Height of resized image.',
    type=click.INT,
    required=False,
)
@click.option(
    '--width',
    help='Width of resized image. If height and width is set, height is prioritized.',
    type=click.INT,
    required=False,
)
@click.option(
    '-t', '--threshold',
    help='Threshold percentage for binarization.',
    type=click.FLOAT,
    required=False,
    default=0.5,
    show_default=True,
)
def pp_cli(**kwargs):
    """
    Preprocess images.

    Binarize images using Kraken, normalize images using Ocropus, and resize images.

    If -r is set, either width or height should be set. If both are set, height is prioritized.

    Multiple FILES can either be passed by absolute paths or by using wildcards.
    """
    preprocess(**kwargs)


@click.group()
@click.help_option('--help', '-h')
@click.version_option(
    '3.0',
    '--version',
    prog_name='Octopy',
    message='\n%(prog)s v%(version)s - Developed at Centre for Philology and Digitality (ZPD), University of Würzburg'
)
def cli(**kwargs):
    pass


cli.add_command(seg_cli)
cli.add_command(segtrain_cli)
cli.add_command(pp_cli)


if __name__ == '__main__':
    """
    Main entry point for pagesegment CLI by calling pagesegment.py directly.
    """
    cli()
