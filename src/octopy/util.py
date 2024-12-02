from typing import Union

from PIL import Image
import numpy as np
import cv2


def kraken_to_string(coords: list[Union[list[int], tuple[int, int]]]) -> str:
    """
    Convert a list of kraken points to a PageXML points string.
    Args:
        coords: List of kraken coords.
    Returns:
        A string containing the points in PageXML format.
    """
    return ' '.join([f"{point[0]},{point[1]}" for point in coords])


def estimate_scales(image: Union[Image.Image, np.ndarray]) -> tuple[int, int]:
    """
    Estimate the median glyph scales of an image.
    Args:
        image: Image to calculate scales.
    Returns:
        A tuple containing the median width and height in pixels.
    """
    if isinstance(image, Image.Image):
        image = np.array(image)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=8)
    if len(stats) > 0:
        avg_width = np.median(stats[1:, 2])  # Average width of the bounding boxes (ignoring the background)
        avg_height = np.median(stats[1:, 3])  # Average height of the bounding boxes
        return max(1, round(avg_width)), max(1, round(avg_height))
    return 25, 25


def is_bitonal(im: Image.Image) -> bool:
    """
    Tests a PIL image for bitonality.
    Args:
        im: PIL Image to test.
    Returns:
        True if the image contains only two different color values. False
        otherwise.
    """
    return im.getcolors(2) is not None and len(im.getcolors(2)) == 2