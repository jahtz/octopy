# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import logging
from os import PathLike
from pathlib import Path
from typing import Literal

from importlib_resources import files
from kraken import blla
from kraken.lib.vgsl import TorchVGSLModel
from kraken.containers import Segmentation, BBoxLine, BaselineLine
from kraken.lib.exceptions import KrakenInvalidModelException
from PIL import Image
from pypxml import PageXML, PageUtil, PageElement, PageType

from octopy.mapping import default_direction_mapping, default_region_mapping


logger: logging.Logger = logging.getLogger(__name__)


class Segmenter:
    """
    Class for segmentation using Kraken.
    """
    
    def __init__(
        self,
        model: PathLike | str | None = None,
        device: str = 'cpu',
        polygonizer: Literal['kraken', 'octopy'] = 'octopy',
        line_fallback_height: int | None = None
    ) -> None:
        self.device = device
        
        if polygonizer == 'octopy':
            try:
                from octopy.plugins import OctopyPolygonizer
                OctopyPolygonizer.register(line_fallback_height)
            except ImportError as exc:
                logger.warning(f'Could not install custom Polygonizer: {str(exc)}')

        self.m: TorchVGSLModel | None = None
        if model:
            try:
                nn = TorchVGSLModel.load_model(model)
                self.m = nn
                if nn.model_type != 'segmentation':
                    raise KrakenInvalidModelException(f'Invalid model type {nn.model_type} for {self.m}')
                if 'class_mapping' not in nn.user_metadata:
                    raise KrakenInvalidModelException(f'Segmentation model {self.m} does not contain valid class mapping')
            except Exception as e:
                logger.error(f'Could not load model ({model}): {e}')
        if self.m is None:
            logger.warning('No custom model passed. Loading default')
            self.m: TorchVGSLModel = TorchVGSLModel.load_model(str(files(blla.__name__) / 'blla.mlmodel'))

    def _res_to_page(
        self, 
        res: Segmentation, 
        creator: str,
        width: int, 
        height: int,
        mode: Literal['lines', 'regions', 'all'] = 'all',
        direction_mapping: dict[str, str] = default_direction_mapping,
        region_mapping: dict[str, tuple[PageType, str | None]] = default_region_mapping
    ) -> PageXML:
        def pts(points: list[tuple[int, int]]) -> str:
            return ' '.join([f'{max(0, p[0])},{max(0, p[1])}' for p in points])
        
        parts: list[str] = Path(res.imagename).name.split('.')
        page = PageXML(
            creator,
            imageFilename=f'{parts[0]}.{parts[-1]}',  # name base + last suffix
            imageWidth=str(width),
            imageHeight=str(height),
            readingDirection=direction_mapping.get(res.text_direction, None)
        )
        if mode == 'lines':
            if res.lines is None:
                return page
            page_region: PageElement = page.create(PageType.TextRegion, type="paragraph", id="r1")
            page_region.create(
                PageType.Coords, 
                points=pts([(0, 0), (width, 0), (width, height), (0, height), (0, 0)])
            )
            for lid, line in enumerate(res.lines, 1):
                page_line: PageElement = page_region.create(PageType.TextLine, id=f'r1_l{lid}')
                if isinstance(line, BBoxLine) and (bbox := line.bbox):
                    xmin, ymin, xmax, ymax = bbox
                    page_line.create(
                        PageType.Coords,
                        points=pts([(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax), (xmin, ymin)])
                    )
                elif isinstance(line, BaselineLine):
                    if boundary := line.boundary:
                        page_line.create(PageType.Coords, points=pts(boundary))
                    if baseline := line.baseline:
                        page_line.create(PageType.Baseline, points=pts(baseline))
            return page

        if res.regions is None:
            return page
        rid = 1
        for region_class, regions in res.regions.items():
            if region_class not in region_mapping:
                logger.warning(f'No mapping for region class: {region_class}')
                continue
            region_type, region_subtype = region_mapping[region_class]
            for region in regions:
                page_region: PageElement = page.create(region_type, type=region_subtype, id=f'r{rid}')
                page_region.create(PageType.Coords, points=pts(region.boundary))
                if mode == 'regions' or res.lines is None:
                    rid += 1
                    continue
                
                lid = 1
                for line in res.lines:
                    if line.regions is not None and region.id in line.regions:
                        page_line: PageElement = page_region.create(PageType.TextLine, id=f'r{rid}_l{lid}')
                        if isinstance(line, BBoxLine) and (bbox := line.bbox):
                            xmin, ymin, xmax, ymax = bbox
                            page_line.create(
                                PageType.Coords,
                                points=pts([(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax), (xmin, ymin)])
                            )
                        elif isinstance(line, BaselineLine):
                            if boundary := line.boundary:
                                page_line.create(PageType.Coords, points=pts(boundary))
                            if baseline := line.baseline:
                                page_line.create(PageType.Baseline, points=pts(baseline))
                        lid += 1
                rid += 1
        return page
    
    def predict(
        self, 
        image: PathLike | str,
        creator: str = 'octopy',
        sort: bool = False,
        mode: Literal['lines', 'regions', 'all'] = 'all',
        text_direction: Literal['horizontal-lr', 'horizontal-rl', 'vertical-lr', 'vertical-rl'] = 'horizontal-lr'
    ) -> PageXML:
        if self.m is None:
            raise ValueError('No model loaded')
        
        im: Image.Image = Image.open(image)
        
        res = blla.segment(
            im=im, 
            text_direction=text_direction, 
            model=self.m, 
            device=self.device,
            # TODO: AUTOCAST?
        )
        page = self._res_to_page(res, creator, im.size[0], im.size[1], mode)
        
        if sort:
            if res.text_direction in ['vertical-lr']:
                direction = 'left-right'
            elif res.text_direction in ['vertical-rl']:
                direction = 'right-left'
            else:
                direction = 'top-bottom'
            PageUtil.sort_regions(page, direction=direction, apply=False)
        
        return page
