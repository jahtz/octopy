from pathlib import Path

import click
import numpy as np
from PIL import Image
from kraken.binarization import nlbin
from kraken.blla import is_bitonal
from scipy.ndimage import percentile_filter, zoom, gaussian_filter, binary_dilation
from scipy import stats


"""
Module: Pre-Processing
Normalize, Binarize or rescale images.
"""


__all__ = ['binarize', 'normalize', 'resize', 'preprocess']


def binarize(image: Image, target: Path | None = None, threshold: float = 0.5) -> Image:
    """
    Binarize PIL Image. Skips processing if image is already binary.

    :param image: Image object.
    :param target: save binarized image to this path. Overrides input image if set to None.
    :param threshold: threshold percentage.
    :return: binarized Image object.
    """
    if not is_bitonal(image):
        image = nlbin(image, threshold)
    if target is not None:
        image.save(target)
    return image


def normalize(image: Image, target: Path | None = None) -> Image:
    """
    Normalize PIL Image.

    :param im: Image object.
    :param target: save normalized image to this path. Overrides input image if set to None.
    :return: normalized Image object.
    """

    # All used functions for normalization are forked from:
    # https://github.com/ocropus-archive/DUP-ocropy/blob/master/ocropus-nlbin
    # and changed to work with Python 3
    # Copyright 2014 Thomas M. Breuel (http://www.apache.org/licenses/LICENSE-2.0)

    def estimate_white_level(image, z=0.5, p=80, s=20) -> Image:
        """
        Flatten PIL Image by estimating the local white level.

        :param image: PIL Image object.
        :param z: zoom for background estimation.
        :param p: percentage for multidimensional percentile filter.
        :param s: x and y size for multidimensional percentile filter.
        :return:
        """
        flat = zoom(image, z)  # zoom in for speed
        flat = percentile_filter(flat, p, size=(s, 2))  # percentile filter x
        flat = percentile_filter(flat, p, size=(2, s))  # percentile filter y
        flat = zoom(flat, (1.0 / z))  # zoom out
        w, h = np.minimum(np.array(image.shape), np.array(flat.shape))
        flat = np.clip(image[:w, :h]-flat[:w, :h]+1, 0, 1)
        return flat
    
    def calculate_thresholds(image, border=0.1, s=1.0, low=5, high=90) -> tuple[float, float]:
        """
        Calculate low and high thresholds.

        :param image: PIL Image object.
        :param border: Ignored border percentile.
        :param s: scale for estimating a mask over the text region.
        :param low: percentile for black elimination.
        :param high: percentile for white elimination.
        :return: tuple of low and high thresholds.
        """
        d0, d1 = image.shape
        o0, o1 = int(border * d0), int(border * d1)
        est = image[o0:d0-o0, o1:d1-o1]
        if s > 0:
            v = est - gaussian_filter(est, (s * 20.0))
            v = gaussian_filter((v ** 2), (s * 20.0)) ** 0.5
            v = (v > 0.3 * np.amax(v))
            v = binary_dilation(v, structure=np.ones((int(s * 50), 1)))
            v = binary_dilation(v, structure=np.ones((1, int(s * 50))))
            est = est[v]
        lo = stats.scoreatpercentile(est.ravel(), low)
        hi = stats.scoreatpercentile(est.ravel(), high)
        return lo, hi
    
    np_image = np.array(image.convert('L'))  # convert image to grayscale and numpy array

    # normalize the image
    np_image = (np_image - np.min(np_image)) / (np.max(np_image) - np.min(np_image))

    # estimate the local white level
    np_image = estimate_white_level(np_image)

    # rescale the image to get the gray scale image
    lo, hi = calculate_thresholds(np_image)
    np_image -= lo
    np_image /= (hi - lo)
    np_image = np.clip(np_image, 0, 1)

    # convert the image back to PIL image
    image = Image.fromarray((np_image * 255).astype(np.uint8))
    if target is not None:
        image.save(target)


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
            if bin:
                binarize(im, output.joinpath(f'{image.name.split(".")[0]}.bin.png'), threshold)
            if nrm:
                normalize(im, output.joinpath(f'{image.name.split(".")[0]}.nrm.png'))