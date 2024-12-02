from typing import Union

import rich_click as click
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


def device_parser(d: str) -> tuple[str, Union[str, list[int]]]:
    """
    Parses the input device string to a pytorch accelerator and device string.
    Args:
        d: Encoded device string (see PyTorch documentation).
    Returns:
        Tuple containing accelerator string and device integer/string.
    """
    auto_devices = ["cpu", "mps"]
    acc_devices = ["cuda", "tpu", "hpu", "ipu"]
    if d in auto_devices:
        return d, "auto"
    elif any([d.startswith(x) for x in acc_devices]):
        dv, i = d.split(':')
        if dv == "cuda":
            dv = "gpu"
        return dv, [int(i)]
    else:
        raise click.BadParameter(f"Invalid device string: {d}")