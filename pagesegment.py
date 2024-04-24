from pathlib import Path

import click
from PIL import Image
from kraken.blla import is_bitonal
from kraken.lib.vgsl import TorchVGSLModel

from modules.transform import image_normalize, image_binarize
from modules.segment import image_segment


def parse_suffix(suffix: str) -> str:
    """ unify suffix format """
    return suffix if suffix.startswith('.') else f'.{suffix}'


@click.command()
@click.help_option("--help", "-h")
@click.version_option(
    "0.1",
    "-v", "--version",
    prog_name="pagesegment",
    message="%(prog)s v%(version)s - Developed at Centre for Philology and Digitality (ZPD), University of Würzburg"
)
@click.argument(
    'files',
    type=click.Path(
        exists=True,
        dir_okay=True,
        file_okay=True,
        resolve_path=True
    ),
    required=True
)
@click.argument(
    'directory',
    type=click.Path(
        exists=False,
        dir_okay=True,
        file_okay=True,
        resolve_path=True
    ),
    required=False
)
@click.option(
    '-r', '--regex',
    help='Regex expression for input file selection. Ignored if FILES points to a single file.',
    type=click.STRING,
    default='*',
    show_default=True
)
@click.option(
    '-B', '--binarize',
    help='Binarize input images and save them to DIRECTORY.',
    type=click.BOOL,
    is_flag=True,
    default=False
)
@click.option(
    '-N', '--normalize',
    help='Normalize input images and save them to DIRECTORY.',
    type=click.BOOL,
    is_flag=True,
    default=False,
)
@click.option(
    '-S', '--segment',
    help='Segment input images with Kraken and output PageXML files to DIRECTORY.',
    type=click.BOOL,
    is_flag=True,
    default=False,
)
@click.option(
    '-b', '--bin_suffix',
    help='Changes output suffix of binarized files, if -B flag is set.',
    type=click.STRING,
    default='.bin.png',
    required=False,
    show_default=True
)
@click.option(
    '-n', '--nrm_suffix',
    help='Changes output suffix of normalized files, if -N flag is set.',
    type=click.STRING,
    default='.nrm.png',
    required=False,
    show_default=True
)
@click.option(
    '-s', '--seg_suffix',
    help='Changes output suffix of PageXML files, if -S flag is set.',
    type=click.STRING,
    default='.xml',
    required=False,
    show_default=True
)
@click.option(
    '--creator',
    help='Set creator of output PageXML file.',
    type=click.STRING,
    default='ZPD Wuerzburg',
    required=False,
    show_default=False
)
@click.option(
    '--scale',
    help='If set, Kraken will recalculate line masks with entered scale factor.',
    type=click.INT,
    required=False,
    show_default=False
)
@click.option(
    '--threshold',
    help='Set binarize threshold in percent.',
    type=click.INT,
    default=50,
    required=False,
    show_default=True
)
@click.option(
    '--device',
    help='Set device for neural network segmentation processing.',
    type=click.STRING,
    default='cpu',
    required=False,
    show_default=True
)
@click.option(
    '--model',
    help='Select Kraken model (.mlmodel) for segmentation. Required for segmentation.',
    type=click.Path(
        exists=True,
        dir_okay=False,
        file_okay=True,
        resolve_path=True
    ),
    required=False,
    show_default=False
)
def cli(files: str, directory: str | None, regex: str,
        binarize: bool, normalize: bool, segment: bool,
        bin_suffix: str, nrm_suffix: str, seg_suffix: str, creator: str,
        scale: int | None, threshold: int, device: str, model: str | None):
    """
    \b
    FILES path to input directory or file.
    DIRECTORY path to output directory, defaults to FILES directory.

    Filenames of output files are identical to input files (cut after first dot!).

    Developed at Centre for Philology and Digitality (ZPD), University of Würzburg.
    """
    # load files
    fp = Path(files)
    fl = [fp] if fp.is_file() else sorted(list(fp.glob(f'{regex}')))  # create list of files
    if len(fl) == 0:
        click.echo(f'No files found in {fp} with regex {regex}!')
        return
    click.echo(f'{len(fl)} files found')

    # set and create output directory
    if directory is None:
        out_dir = fp.parent if fp.is_file() else fp
    else:
        out_dir = Path(directory)
        out_dir.mkdir(parents=True, exist_ok=True)

    # load model
    m = None
    if model is not None:
        m = TorchVGSLModel.load_model(model)
        click.echo('Model loaded')

    with click.progressbar(fl, label='Processing files', show_pos=True, show_eta=True, show_percent=True,
                           item_show_func=lambda x: f'{x.name} ' if x is not None else ' ') as images:
        for image_fp in images:
            image = Image.open(image_fp)  # load image
            name_base = image_fp.name.split('.')[0]  # get filename without suffix

            # normalize image
            if normalize:
                image_normalize(image, out_dir.joinpath(f'{name_base}{parse_suffix(nrm_suffix)}'))

            # binarize if not bitonal
            bt = is_bitonal(image)
            if binarize and bt:
                image.save(out_dir.joinpath(f'{name_base}{parse_suffix(bin_suffix)}'))
            elif not bt and binarize:
                image = image_binarize(image, threshold=(threshold / 100.0),
                                       out_path=out_dir.joinpath(f'{name_base}{parse_suffix(bin_suffix)}'))
            elif not bt and not binarize and segment:
                image = image_binarize(image, threshold=(threshold / 100.0))

            # segment image
            if segment:
                if model is None:
                    click.echo('No model provided for segmentation!')
                    return
                image_segment(image, image_fp.name, out_dir.joinpath(f'{name_base}{parse_suffix(seg_suffix)}'),
                              creator=creator, model=m, device=device, scale=scale)

    click.echo('Done!')
