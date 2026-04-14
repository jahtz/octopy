# SPDX-License-Identifier: Apache-2.0
import logging
from pathlib import Path
from typing import Literal

from kraken.configs import SegmentationInferenceConfig
from kraken.containers import Segmentation, BBoxLine, BaselineLine
from kraken.ketos.util import to_ptl_device
from kraken.tasks import SegmentationTaskModel
from kraken.models.utils import create_model
from PIL import Image
from pypxml import PageXML, PageType, PageElement, PageUtil

from .util import spinner, progressbar
from .mappings import default_direction_mapping, default_region_mapping


logger: logging.Logger = logging.getLogger(__name__)


def segmentation_to_pagexml(
    res: Segmentation,
    image_width: int,
    image_height: int,
    creator: str,
    mode: Literal['lines', 'regions', 'all'] = 'all',
    sort: bool = False,
    direction_mapping: dict[str, str] = default_direction_mapping,
    region_mapping: dict[str, tuple[PageType, str | None]] = default_region_mapping
) -> PageXML:
    def pts(points: list[tuple[int, int]]) -> str:
        return ' '.join([f'{p[0]},{p[1]}' for p in points])
    
    parts: list[str] = Path(res.imagename).name.split('.')
    page = PageXML(
        creator,
        imageFilename=f'{parts[0]}.{parts[-1]}',  # name base + last suffix
        imageWidth=str(image_width),
        imageHeight=str(image_height),
        readingDirection=direction_mapping.get(res.text_direction, None)
    )
    if mode == 'lines':
        if res.lines is None:
            return page
        page_region: PageElement = page.create(PageType.TextRegion, type="paragraph", id="r1")
        page_region.create(
            PageType.Coords, 
            points=pts([(0, 0), (image_width, 0), (image_width, image_height), (0, image_height), (0, 0)])
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
    if sort:            
        if res.text_direction in ['vertical-lr']:
            direction = 'left-right'
        elif res.text_direction in ['vertical-rl']:
            direction = 'right-left'
        else:
            direction = 'top-bottom'
        PageUtil.sort_regions(page, direction=direction, apply=False)
    return page


def segment(
    images: list[Path] | Path,
    model: Path | None = None,
    output: Path | None = None,
    output_suffix: str = '.xml',
    text_direction: Literal[
        'horizontal-lr', 
        'horizontal-rl', 
        'vertical-lr', 
        'vertical-rl'
    ] = 'horizontal-lr',
    precision: Literal[
        'transformer-engine', 
        'transformer-engine-float16', 
        '16-true', 
        '16-mixed', 
        'bf16-true', 
        'bf16-mixed', 
        '32-true', 
        '64-true'
    ] = '32-true',
    device: str = 'auto',
    threads: int = 1,
    creator: str = 'octopy',
    mode: Literal['lines', 'regions', 'all'] = 'all',
    sort: bool = False
) -> None:
    """
    Segment images using Kraken segmentation module.

    Args:
        images: The image(s) to segment.
        model Custom segmentation model. If no model is provided, the default Kraken blla model is used. 
            Defaults to None.
        output: Custom output directory for created PAGE-XML files. If not set, the parent directory of each input 
            image is used. Defaults to None.
        output_suffix: Full extension of the created PAGE-XML files. Defaults to '.xml'.
        text_direction: Principal text direction. Options: 'horizontal-lr', 'horizontal-rl', 'vertical-lr', 
            'vertical-rl'. Defaults to 'horizontal-lr'.
        precision: Numerical precision to use for inference. Options: 'transformer-engine', 
            'transformer-engine-float16', '16-true', '16-mixed', 'bf16-true', 'bf16-mixed', '32-true', '64-true'.
            Defaults to '32-true'.
        device: Specify the processing device (e.g. 'cpu', 'cuda:0',...). Refer to PyTorch documentation for supported 
            devices. Defaults to 'auto'.
        threads: Maximum size of OpenMP/BLAS thread pool. Defaults to 1.
        creator: Custom PAGE-XML metadata creator string. Defaults to 'octopy'.
        mode: Set segmentation mode. Options: 'lines', 'regions', 'all'. Defaults to 'all'.
        sort: Sort the regions using the model specifications.
    """
    if isinstance(images, Path):
        images: list[Path] = [images]
    
    a, d = to_ptl_device(device)
    config = SegmentationInferenceConfig(
        text_direction=text_direction,
        num_threads=threads,
        precision=precision,
        accelerator=a,
        device=d
    )
    
    with spinner as sp:
        sp.add_task('Loading model', total=None)
        m = create_model('OctopySegmentationModel', weights=model, bbox_pad=0)
        segmenter = SegmentationTaskModel([m])
        #segmenter: SegmentationTaskModel = SegmentationTaskModel.load_model(model)

    with progressbar as pb:
        task = pb.add_task('', total=len(images), status='')
        for fp in images:
            pb.update(task, status='/'.join(fp.parts[-4:]))
            try:
                im: Image.Image = Image.open(fp)
                width, height = im.size
                
                res: Segmentation = segmenter.predict(im, config)  # line throwaway on error: kraken/lib/vgls/spred.py:146
                page: PageXML = segmentation_to_pagexml(res, width, height, creator, mode, sort)
                
                if output:
                    out: Path = output.joinpath(fp.name.split('.')[0] + output_suffix)
                else:
                    out: Path = fp.parent.joinpath(fp.name.split('.')[0] + output_suffix)
                    
                page.save(out)
            except Exception as ex:
                logger.error(f'Cloud not segment image {fp.as_posix()}: {ex}')
            pb.advance(task)
        pb.update(task, status='Done')
