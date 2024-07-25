from pathlib import Path

import click
from PIL import Image
from kraken.binarization import nlbin


"""
Module: Pre-Processing
Normalize, Binarize or rescale images.
"""


__all__ = ['resize', 'preprocess']


def resize(image: Image, target: Path | None = None, height: int | None = None, width: int | None = None) -> Image:
    """
    Resize PIL Image.

    :param image: Image object.
    :param target: save resized image to this path. Overrides input image if set to None.
    :param height: height of resized image.
    :param width: width of resized image.
    :return: resized Image object.
    """
    if height is None and width is None:
        raise ValueError('Either height or width must be specified.')
    
    original_width, original_height = image.size
    aspect_ratio = original_width / original_height

    if width is None:
        new_width = int(height * aspect_ratio)
        image = image.resize((new_width, height), Image.LANCZOS)

    else:
        new_height = int(width / aspect_ratio)
        image = image.resize((width, new_height), Image.LANCZOS)

    if target is not None:
        image.save(target)
    return image


def preprocess(
    files: list[Path], 
    output: Path,
    bin: bool = False,
    nrm: bool = False,
    res: bool = False,
    height: int | None = None,
    width: int | None = None,
    threshold: float = 0.5,
):
    """
    Pre-process a set of images.

    :param files: list of image files to pre-process.
    :param output: output directory to save the pre-processed files.
    :param binarize: binarize images.
    :param normalize: normalize images.
    :param resize: resize images. Used for binarization and normalization.
    :param height: height of resized image.
    :param width: width of resized image. If height and width is set, height is prioritized.
    :param threshold: threshold percentage for binarization.
    """
    output.mkdir(parents=True, exist_ok=True)
    with click.progressbar(files, label='Preprocess files', show_pos=True, show_eta=True, show_percent=True,
                           item_show_func=lambda f: f.name if f is not None else '') as images:
        for image in images:
            im = Image.open(image)
            if res:
                im = resize(im, None, height, width)
            if bin or nrm:
                im_bin, im_nrm = nlbin(image, threshold)
                if bin:
                    im_bin.save(output.joinpath(f'{image.name.split(".")[0]}.bin.png'))
                if nrm:
                    im_nrm.save(output.joinpath(f'{image.name.split(".")[0]}.nrm.png'))
                    