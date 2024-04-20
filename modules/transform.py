from pathlib import Path

import numpy as np
from kraken.binarization import nlbin
from PIL import Image
from scipy.ndimage import percentile_filter, zoom, gaussian_filter, binary_dilation
from scipy import stats


def nrm_white_level(image, z=0.5, p=80, s=20) -> Image:
    """
    Flatten PIL Image by estimating the local white level.

    :param image: PIL Image object.
    :param z: zoom for background estimation.
    :param p: percentage for multidimensional percentile filter.
    :param s: x and y size for multidimensional percentile filter.
    :return:
    """
    # Forked from DUP-Ocropy.nlbin and changed to work with Python 3
    # Copyright 2014 Thomas M. Breuel
    # http://www.apache.org/licenses/LICENSE-2.0
    flat = zoom(image, z)  # zoom in for speed
    flat = percentile_filter(flat, p, size=(s, 2))  # percentile filter x
    flat = percentile_filter(flat, p, size=(2, s))  # percentile filter y
    flat = zoom(flat, (1.0 / z))  # zoom out
    w, h = np.minimum(np.array(image.shape), np.array(flat.shape))
    flat = np.clip(image[:w, :h]-flat[:w, :h]+1, 0, 1)
    return flat


def nrm_thresholds(image, border=0.1, s=1.0, low=5, high=90) -> tuple[float, float]:
    """
    Calculate low and high thresholds.

    :param image: PIL Image object.
    :param border: Ignored border percentile.
    :param s: scale for estimating a mask over the text region.
    :param low: percentile for black elimination.
    :param high: percentile for white elimination.
    :return: tuple of low and high thresholds.
    """
    # Forked from DUP-Ocropy.nlbin and changed to work with Python 3
    # Copyright 2014 Thomas M. Breuel
    # http://www.apache.org/licenses/LICENSE-2.0
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


def image_normalize(im, out_path: Path | None = None) -> Image:
    """ Normalizes image """
    # convert image to grayscale and numpy array
    gs_image = np.array(im.convert('L'))

    # normalize the image
    nrm_img = (gs_image - np.min(gs_image)) / (np.max(gs_image) - np.min(gs_image))

    # estimate the local white level
    wl_img = nrm_white_level(nrm_img)

    # rescale the image to get the gray scale image
    lo, hi = nrm_thresholds(wl_img)
    wl_img -= lo
    wl_img /= (hi - lo)
    sc_img = np.clip(wl_img, 0, 1)

    # convert the image back to PIL image
    im_norm = Image.fromarray((sc_img * 255).astype(np.uint8))
    if out_path is not None:
        im_norm.save(out_path)


def image_binarize(im: Image, out_path: Path | None = None, threshold: float = 0.5) -> Image:
    """ Binarizes image """
    im_bin = nlbin(im, threshold)
    if out_path is not None:
        im_bin.save(out_path)
    return im_bin
