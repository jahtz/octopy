# Copyright 2024 Janik Haitz
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from typing import Union, Optional
from pathlib import Path

from rich import print as rprint
from pypxml import PageXML, PageType
from shapely.geometry import Polygon, LineString
from shapely.ops import unary_union
from scipy.spatial import distance_matrix
import numpy as np
import cv2

from util import from_points, to_points, mask_image, estimate_scales, region_contours


def validate_polygons(polygons: list[Polygon]) -> Optional[Polygon]:
    """
    Validate a list of polygons.
    Args:
        polygons: List of polygons to validate.
    Returns:
        A single polygon.
    """
    # merge overlapping geometries
    merged_geom = unary_union(polygons)

    # if already a single polygon, return it
    if merged_geom.geom_type == "Polygon":
        return merged_geom
    else:
        polygons = list[merged_geom]
        centroids = np.array([poly.centroid.coords[0] for poly in polygons])

        # compute distance matrix (pairwise distances between polygon centroids)
        dist_matrix = distance_matrix(centroids, centroids)
        np.fill_diagonal(dist_matrix, np.inf)  # avoid self-connections

        # compute minimum spanning tree (Prim's algorithm)
        n = len(polygons)
        in_tree = [False] * n
        in_tree[0] = True
        mst_edges = []
        for _ in range(n - 1):
            min_dist = np.inf
            min_edge = None
            for i in range(n):
                if in_tree[i]:
                    for j in range(n):
                        if not in_tree[j] and dist_matrix[i, j] < min_dist:
                            min_dist = dist_matrix[i, j]
                            min_edge = (i, j)
            if min_edge:
                mst_edges.append(min_edge)
                in_tree[min_edge[1]] = True

        # add connecting edges
        connecting_lines = [LineString([centroids[i], centroids[j]]) for i, j in mst_edges]

        # merge polygons and lines into a single geometry
        combined_geometry = unary_union(polygons + connecting_lines)

        # valid resulting polygon
        if combined_geometry.geom_type == "Polygon":
            return combined_geometry
        elif combined_geometry.geom_type == "MultiPolygon":
            all_exteriors = []
            for poly in combined_geometry:
                all_exteriors.extend(poly.exterior.coords)
            return Polygon(all_exteriors)
        else:
            return None


def region_shrink(pagexml: Union[PageXML, Path, str],
                  image: Union[Path, str],
                  padding: int = 5,
                  h_smoothing: int = 1,
                  v_smoothing: int = 1,
                  valid_regions: Optional[list[str]] = None) -> PageXML:
    """
    Shrink PageXML regions to its content.
    Args:
        pagexml: The PageXML file to modify.
        image: Matching image file.
        padding: padding between the shrunk region and its content.
        h_smoothing: The higher, the more horizontal smoothing is applied.
        v_smoothing: The higher, the more vertical smoothing is applied.
        valid_regions: A list of valid region types to shrink. If nothing is provided, all regions are shrunk.
    Returns:
        The modified PageXML object.
    """
    if isinstance(pagexml, Path) or isinstance(pagexml, str):
        pagexml = PageXML.from_xml(pagexml)
    inverted_image = ~ cv2.imread(image, cv2.IMREAD_GRAYSCALE)

    # check, if page size matches image size
    h, w = inverted_image.shape
    if h != pagexml.height or w != pagexml.width:
        rprint("[bold red]ERROR:[/bold red] Image Size does not match the page size specified in the PageXML file.")
        raise ValueError

    for region in pagexml.regions:
        if valid_regions and region.type.value not in valid_regions:
            continue  # filters regions by type if valid_regions is provided
        coords_element = region.get_coords()
        if coords_element is None:
            rprint(f"[bold orange]WARNING:[/bold orange] Could not find Coords element in region {region.id}")
            continue
        region_polygon = np.array(from_points(coords_element["points"]), dtype=np.int32)
        masked_image = mask_image(inverted_image, region_polygon)

        if region.type in [PageType.TextRegion, PageType.SeparatorRegion]:
            glyph_width, glyph_height = estimate_scales(masked_image)
        else:
            glyph_width, glyph_height = 25, 25

        # dilate text
        dilation_kernel = np.ones((glyph_width, glyph_width), np.uint8)
        dilated_image = cv2.dilate(masked_image, dilation_kernel, iterations=2)

        # merge symbols
        smooth_kernel = np.ones((glyph_height * v_smoothing, glyph_width * h_smoothing), np.uint8)
        smoothed_image = cv2.morphologyEx(dilated_image, cv2.MORPH_CLOSE, smooth_kernel)

        contours = region_contours(smoothed_image)
        areas = [cv2.contourArea(contour) for contour in contours]
        if not sum(areas):
            rprint(f"[bold orange]WARNING:[/bold orange] Shrunk region area is 0 in region {region.id}")
            continue

        polys = [Polygon([tuple(pt[0]) for pt in contour]) for contour in contours]
        polys = [poly.buffer(padding - glyph_width, join_style=2) for poly in polys]
        polys = [poly for poly in polys if poly.area > 0]
        if polys:
            try:
                poly = validate_polygons(polys)
                if poly is None:
                    print(f'Could not merge multiple resulting polygons in region `{region.id}`.')
                    continue
                arr = np.array(poly.exterior.coords, dtype=np.int32)
                coords_element['points'] = to_points(arr)
            except AssertionError as e:
                rprint(f'Could not merge multiple resulting polygons in region `{region.id}`. ({e})')
        else:
            rprint(f'Could not shrink region `{region.id}`.')
    return pagexml
