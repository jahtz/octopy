from typing import Optional

from pagexml import Element, ElementType
from pagexml.geometry import Polygon

"""
Module: Postprocessing
Postprocessing methods
"""

def pxml_remove_empty_regions(regions: list[Element], inplace: bool = False) -> Optional[list[Element]]:
    """
    Deletes a region if it contains no elements except the mask coords.
    """
    if not inplace:
        regions = regions.copy()
    for region in regions:
        if len(region) <= 1:
            regions.remove(region)
    return None if inplace else regions


def pxml_sort_regions(regions: list[Element], inplace: bool = False) -> Optional[list[Element]]:
    """
    Sorts a list of TextRegion Elements by its center x coordinate (left to right).
    Inplace sorting.
    """
    if not inplace:
        regions = regions.copy()
    regions.sort(key=lambda r: -1 if r.etype != ElementType.TextRegion else Polygon.from_page_coords(r.get_coords_element()['points']).center().x)
    return None if inplace else regions


def pxml_sort_lines(lines: list[Element], inplace: bool = False) -> Optional[list[Element]]:
    """
    Sorts a list of TextLine Elements by its center y coordinate (top to bottom).
    """
    if not inplace:
        lines = lines.copy()
    lines.sort(key=lambda l: -1 if l.etype != ElementType.TextLine else Polygon.from_page_coords(l.get_coords_element()['points']).center().y)
    return None if inplace else lines


def pxml_merge_overlapping_regions(regions: list[Element]) -> list[Element]:
    """
    Merges overlapping regions and merge its content. Sort lines after merge.
    """
    # TODO: work in progress
    return regions

