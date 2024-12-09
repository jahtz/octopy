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


def to_points(coords: list[Union[list[int], tuple[int, int]]]) -> str:
    """
    Build a PageXML coordinate string from a list of sublists/tuples containing x,y coordinates
    Args:
        coords: List of x,y coordinates, where each coordinate is a tuple or list.
    Returns:
        PageXML coords string of type `x1,y1 x2,y2 ... xn,yn`
    """
    return ' '.join([f"{point[0]},{point[1]}" for point in coords])

def from_points(points: str) -> list[tuple[int, int]]:
    """
    Parse a PageXML coordinate string to a list of tuples containing x,y coordinates.
    Args:
        points: PageXML coordinate points string of type `x1,y1 x2,y2 ... xn,yn`
    Returns:
        List of tuples containing x,y coordinates.
    """
    polygon = []
    for pair in points.split(' '):
        x_y = pair.split(',')
        polygon.append((int(x_y[0]), int(x_y[1])))
    return polygon


def mask_image(image: np.ndarray, mask_poly: np.ndarray) -> np.ndarray:
    """
    Mask an image with a polygon.
    Args:
        image: The image to mask.
        mask_poly: The polygon to mask the image with.
    Returns:
        The masked image.
    """
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [mask_poly], (255, 255, 255))
    return image & mask


def region_contours(masked_image: np.ndarray) -> list[np.ndarray]:
    """
    Get the contours of a masked image.
    Args:
        masked_image: The masked image to extract contours from.
    Returns:
        A list of contours sorted by area in descending order
    """
    contours = cv2.findContours(masked_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    return sorted(contours, key=cv2.contourArea, reverse=True)
