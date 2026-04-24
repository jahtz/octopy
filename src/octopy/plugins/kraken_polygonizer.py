# SPDX-License-Identifier: Apache-2.0
"""
This plugin fixes Kraken's default polygonizer behavior.

In Kraken's default implementation, if polygonization fails for a line, that
line is discarded entirely. This plugin preserves such lines by falling
back to a simple bounding box with a predefined height.
"""
from __future__ import annotations

import logging
from typing import Any

from collections.abc import Sequence

from kraken.lib.segmentation import _calc_roi, _extract_patch
from kraken.lib.vgsl import spred
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
import shapely.geometry as geom
from skimage.filters import sobel


logger: logging.Logger = logging.getLogger('octopy')


def _calculate_fallback(line: list[tuple[int, int]], fraction: float) -> list[tuple[int, int]] | None:
    y: int = round(sum(p[1] for p in line) / len(line))
    return [
        (line[0][0], y - round(KrakenPolygonizer.FALLBACK_HEIGHT * fraction)),
        (line[-1][0], y - round(KrakenPolygonizer.FALLBACK_HEIGHT * fraction)),
        (line[-1][0], y + round(KrakenPolygonizer.FALLBACK_HEIGHT * (1 - fraction))),
        (line[0][0], y + round(KrakenPolygonizer.FALLBACK_HEIGHT * (1-fraction)))
    ]
    

def calculate_polygonal_environment(
    im: Image.Image = None,  # ty:ignore[invalid-parameter-default]
    baselines: Sequence[Sequence[tuple[int, int]]] = None,  # ty:ignore[invalid-parameter-default]
    suppl_obj: Sequence[Sequence[tuple[int, int]]] = None,  # ty:ignore[invalid-parameter-default]
    im_feats: np.ndarray = None,  # ty:ignore[invalid-parameter-default]
    scale: tuple[int, int] = None,  # ty:ignore[invalid-parameter-default]
    topline: bool | None = False,
    raise_on_error: bool = False
) -> Any:
    """
    Forked from the original calculate_polygonal_environment method in kraken/lib/segmentation.py
    """
    if scale is not None and (scale[0] > 0 or scale[1] > 0):
        w, h = im.size
        oh, ow = scale
        if oh == 0:
            oh = int(h * ow / w)
        elif ow == 0:
            ow = int(w * oh / h)
        im = im.resize((ow, oh))
        scale = np.array((ow / w, oh / h))  # ty:ignore[invalid-assignment]
        # rescale baselines
        baselines = [(np.array(bl) * scale).astype('int').tolist() for bl in baselines]
        if suppl_obj is not None:
            suppl_obj = [(np.array(bl) * scale).astype('int').tolist() for bl in suppl_obj]

    if im_feats is None:
        bounds = np.array(im.size, dtype=float) - 1
        im = np.array(im.convert('L'))
        # compute image gradient
        im_feats = gaussian_filter(sobel(im), 0.5)
    else:
        bounds = np.array(im_feats.shape[::-1], dtype=float) - 1

    polygons = []
    if suppl_obj is None:
        suppl_obj = []

    for idx, line in enumerate(baselines):
        try:
            end_points = (line[0], line[-1])
            line = geom.LineString(line)
            offset = 4 if topline is not None else 0
            offset_line = line.parallel_offset(offset, side='left' if topline else 'right')
            line = np.array(line.coords, dtype=float)
            offset_line = np.array(offset_line.coords, dtype=float)

            # calculate magnitude-weighted average direction vector
            lengths = np.linalg.norm(np.diff(line.T), axis=0)
            p_dir = np.mean(np.diff(line.T) * lengths / lengths.sum(), axis=1)
            p_dir = (p_dir.T / np.sqrt(np.sum(p_dir**2, axis=-1)))
            env_up, env_bottom = _calc_roi(
                line, 
                bounds, 
                baselines[:idx] + baselines[idx + 1:],  # ty:ignore[unsupported-operator]
                suppl_obj, 
                p_dir
            )  

            polygons.append(_extract_patch(
                env_up,
                env_bottom,
                line.astype('int'),
                offset_line.astype('int'),
                end_points,
                p_dir,
                topline,
                offset,
                im_feats,
                bounds
            ))
        except Exception as e:
            if raise_on_error:
                raise
            if KrakenPolygonizer.FALLBACK_HEIGHT:
                logger.info(f'Polygonizer failed on line {idx}: {e}. Calculate fallback polygon.')
                if topline is None:  # centerline
                    polygons.append(_calculate_fallback(line, 1/2))  # ty:ignore[invalid-argument-type]
                elif topline:  # topline
                    polygons.append(_calculate_fallback(line, 1/3))  # ty:ignore[invalid-argument-type]
                else:  # baseline
                    polygons.append(_calculate_fallback(line, 2/3))  # ty:ignore[invalid-argument-type]
            else:               
                logger.warning(f'Polygonizer failed on line {idx}: {e}. Omitting line.')
                polygons.append(None)
            
    if scale is not None:
        polygons = [(np.array(pol) / scale).astype('uint').tolist() if pol is not None else None for pol in polygons]
    return polygons


class KrakenPolygonizer:
    FALLBACK_HEIGHT: int
    
    @staticmethod
    def register(fallback_height: int) -> None:
        KrakenPolygonizer.FALLBACK_HEIGHT: int | None = fallback_height
        spred.calculate_polygonal_environment = calculate_polygonal_environment  # ty:ignore[invalid-assignment]
        logger.info('Plugin: KrakenPolygonizer registered')
