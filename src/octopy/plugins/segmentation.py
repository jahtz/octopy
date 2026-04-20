# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from collections.abc import Sequence
from importlib import import_module
import logging
from os import getenv

from kraken.lib.segmentation import _calc_roi, _extract_patch
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
import shapely.geometry as geom
from skimage.filters import sobel


logger: logging.Logger = logging.getLogger('octopy')


v: str | None = getenv('OCTOPY_SEGMENTATION_FALLBACK')
if v is None or v == '' or v.lower() in {'none', 'null'}:
    FALLBACK = None
else:
    FALLBACK = int(v)


def calculate_fallback(line: list[tuple[int, int]], fraction: float) -> list[tuple[int, int]] | None:
    if FALLBACK is None:
        return None
    y = int(sum(p[1] for p in line) / len(line))
    return [
        (line[0][0], y - int(FALLBACK * fraction)),
        (line[-1][0], y - int(FALLBACK * fraction)),
        (line[-1][0], y + int(FALLBACK * (1 - fraction))),
        (line[0][0], y + int(FALLBACK * (1-fraction)))
    ]


def register() -> None:
    """
    Kraken plugin entrypoint
    """
    segmentation = import_module('kraken.lib.segmentation')
    spred = import_module('kraken.lib.vgsl.spred')
    if not hasattr(segmentation, 'calculate_polygonal_environment'):
        logger.warning('segfix: kraken.lib.segmentation.calculate_polygonal_environment not found')
        return
    
    if not hasattr(spred, 'calculate_polygonal_environment'):
        logger.warning('segfix: kraken.lib.vgsl.spred.calculate_polygonal_environment not found')
        return

    original_calculate_polygonal_environment = segmentation.calculate_polygonal_environment
    
    def calculate_polygonal_environment(
        im: Image.Image | None = None,
        baselines: Sequence[Sequence[tuple[int, int]]] | None = None,
        suppl_obj: Sequence[Sequence[tuple[int, int]]] | None = None,
        im_feats: np.ndarray | None = None,
        scale: tuple[int, int] | None = None,
        topline: bool | None = False,
        raise_on_error: bool = False
    ):
        """
        Forked from the original calculate_polygonal_environment method in kraken/lib/segmentation.py
        """
        if scale is not None and (scale[0] > 0 or scale[1] > 0):
            w, h = im.size  # ty:ignore[unresolved-attribute]
            oh, ow = scale
            if oh == 0:
                oh = int(h * ow / w)
            elif ow == 0:
                ow = int(w * oh / h)
            im = im.resize((ow, oh))  # ty:ignore[unresolved-attribute]
            scale = np.array((ow / w, oh / h))  # ty:ignore[invalid-assignment]
            # rescale baselines
            baselines = [(np.array(bl) * scale).astype('int').tolist() for bl in baselines]  # ty:ignore[not-iterable]
            # rescale suppl_obj
            if suppl_obj is not None:
                suppl_obj = [(np.array(bl) * scale).astype('int').tolist() for bl in suppl_obj]

        if im_feats is None:
            bounds = np.array(im.size, dtype=float) - 1  # ty:ignore[unresolved-attribute]
            im = np.array(im.convert('L'))  # ty:ignore[invalid-assignment, unresolved-attribute]
            # compute image gradient
            im_feats = gaussian_filter(sobel(im), 0.5)
        else:
            bounds = np.array(im_feats.shape[::-1], dtype=float) - 1

        polygons = []
        if suppl_obj is None:
            suppl_obj = []

        for idx, line in enumerate(baselines):  # ty:ignore[invalid-argument-type]
            try:
                end_points = (line[0], line[-1])  # ty:ignore[not-subscriptable]
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
                    baselines[:idx] + baselines[idx + 1:],  # ty:ignore[unsupported-operator, not-subscriptable]
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
                if FALLBACK:
                    logger.info(f'Polygonizer failed on line {idx}: {e}. Calculate fallback polygon.')
                    if topline is None:  # centerline
                        polygons.append(calculate_fallback(line, 1/2))  # ty:ignore[invalid-argument-type]
                    elif topline:  # topline
                        polygons.append(calculate_fallback(line, 1/3))  # ty:ignore[invalid-argument-type]
                    else:  # baseline
                        polygons.append(calculate_fallback(line, 2/3))  # ty:ignore[invalid-argument-type]
                else:               
                    logger.warning(f'Polygonizer failed on line {idx}: {e}. Omitting line.')
                    polygons.append(None)
                
        if scale is not None:
            polygons = [(np.array(pol) / scale).astype('uint').tolist() if pol is not None else None for pol in polygons]
        return polygons
    

    segmentation.calculate_polygonal_environment = calculate_polygonal_environment  # ty:ignore[invalid-assignment]
    spred.calculate_polygonal_environment = calculate_polygonal_environment  # ty:ignore[invalid-assignment]
    setattr(segmentation, '__segfix_original_calculate_polygonal_environment', original_calculate_polygonal_environment)
    logger.info('segfix: replaced kraken.lib.segmentation.calculate_polygonal_environment and kraken.lib.vgsl.spred.calculate_polygonal_environment')
