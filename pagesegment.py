import click

from modules import *


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
    '--scale',
    help='Set Kraken segmentation scale factor. Higher value -> increased computation time.',
    type=click.INT,
    default=1200,
    required=False,
    show_default=True
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
    '--threads',
    help='Set thread count for processing.',
    type=click.INT,
    default=1,
    required=False,
    show_default=True
)
def cli(files: str, directory: str | None, regex: str,
        binarize: bool, normalize: bool, segment: bool,
        bin_suffix: str, nrm_suffix: str, seg_suffix: str,
        scale: int, threshold: int, threads: int):
    """
    \b
    FILES path to input directory or file.
    DIRECTORY path to output directory, defaults to FILES directory.

    Filenames of output files are identical to input files (cut after first dot!).

    Developed at Centre for Philology and Digitality (ZPD), University of Würzburg.
    """


if __name__ == '__main__':
    cli()