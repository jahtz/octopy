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

import logging
import inspect
from pathlib import Path
from typing import Optional, Union, Literal

from importlib_resources import files
import lightning as _  # fixes "Segmentation Fault (Core dumped)"
from kraken import blla
from kraken.lib.vgsl import TorchVGSLModel
from kraken.containers import Segmentation
from kraken.lib.exceptions import KrakenInvalidModelException
import numpy as np
from PIL import Image
from pypxml import PageXML, PageType

from . import util

TEXT_DIRECTION = Literal["hlr", "hrl", "vlr", "vrl"]
CUSTOM_KRAKEN = len(inspect.signature(blla.segment).parameters) > 8  # check if the modified kraken version is installed
log = logging.getLogger("octopy")


def segmentation_to_page(
    res: Segmentation,
    image_width: int,
    image_height: int,
    creator: str,
    suppress_lines: bool = False,
    suppress_regions: bool = False
) -> PageXML:
    """
    Convert a kraken segmentation result to a PageXML object.
    Args:
        res: The result of the kraken blla segmentation.
        image_width: Width of the input image in pixels.
        image_height: Height of the input image in pixels.
        creator: Creator metadata for the PageXML file.
        suppress_lines: Suppress lines in the output PageXML.
        suppress_regions: Suppress regions in the output PageXML.
            Creates a single dummy region for the whole image.
    Returns:
        A PageXML object containing the segmentation result.
    """
    fn = Path(res.imagename).name.split('.')
    pagexml = PageXML(
        creator=creator,
        imageFilename=f"{fn[0]}.{fn[-1]}",  # name base + last suffix
        imageHeight=image_height,
        imageWidth=image_width
    )
    rc, lc = 1, 1
    if suppress_regions:
        dummy_region = pagexml.create_element(PageType.TextRegion, type="paragraph", id="rdummy")
        dummy_region.create_element(
            PageType.Coords, 
            points=util.points_to_string([(0, 0), (image_width, 0), (image_width, image_height), (0, image_height)])
        )
        if suppress_lines:
            return pagexml
        for line in res.lines:
            textline = dummy_region.create_element(PageType.TextLine, id=f"l{lc}")
            if boundary := line.boundary:
                textline.create_element(PageType.Coords, points=util.points_to_string(boundary))
            if baseline := line.baseline:
                textline.create_element(PageType.Baseline, points=util.points_to_string(baseline))
            lc += 1
    else:
        for region_class, found_regions in res.regions.items():
            if region_class not in util.SEGMENTATION_CLASS_MAPPING:
                log.warning(f"Unknown region class: {region_class}")
                continue
            pagetype, type_attribute = util.SEGMENTATION_CLASS_MAPPING[region_class]
            for found_region in found_regions:
                region = pagexml.create_element(pagetype, type=type_attribute, id=f"r{rc}")
                region.create_element(PageType.Coords, points=util.points_to_string(found_region.boundary))
                if suppress_lines:
                    continue
                for found_line in res.lines:
                    if found_region.id in found_line.regions:
                        textline = region.create_element(PageType.TextLine, id=f"l{lc}")
                        if boundary := found_line.boundary:
                            textline.create_element(PageType.Coords, points=util.points_to_string(boundary))
                        if baseline := found_line.baseline:
                            textline.create_element(PageType.Baseline, points=util.points_to_string(baseline))
                        lc += 1
                rc += 1
    return pagexml


def segment(
    images: Union[Path, list[Path]],
    models: Optional[list[Path]] = None,
    output: Optional[Path] = None,
    suffix: str = ".xml",
    device: str = "cpu",
    creator: str = "octopy",
    text_direction: TEXT_DIRECTION = "hlr",
    suppress_lines: bool = False,
    suppress_regions: bool = False,
    fallback_polygon: Optional[int] = None,
    heatmap: Optional[str] = None
) -> None:
    """
    Segment images using Kraken.
    Args:
        images: Image path or a set of image paths.
        models: Path to custom segmentation model(s). If set to None, the default Kraken model is used.
        output: Output directory for processed files. Defaults to the parent directory of each input file.
        output_suffix: Suffix for output PageXML files. Should end with '.xml'.
        device: Specify the processing device (e.g. 'cpu', 'cuda:0',...).
            Refer to PyTorch documentation for supported devices.
        creator: Metadata: Creator of the PageXML files.
        text_direction: Text direction of input images.
        suppress_lines: Suppress lines in the output PageXML.
        suppress_regions: Suppress regions in the output PageXML. Creates a single dummy region for the whole image.
        fallback_polygon: Use a default bounding box when the polygonizer fails to create a polygon around a baseline.
            Requires a box height in pixels.
        heatmap: Generate a heatmap image alongside the PageXML output.
            Specify the file extension for the heatmap (e.g., `.hm.png`).
    """
    if not CUSTOM_KRAKEN:
        log.warning("Some features are not available due to the installed Kraken version")
        
    if not isinstance(images, list):
        images = [images]

    # Load models
    torch_model = []
    with util.SPINNER as spinner:
        spinner.add_task(description="Loading models", total=None)
        for model in models:
            try:
                nn = TorchVGSLModel.load_model(model)
                torch_model.append(nn)
            except Exception as e:
                log.error(f"Could not load model ({model}): {e}")
                continue
            if nn.model_type != "segmentation":
                raise KrakenInvalidModelException(f"Invalid model type {nn.model_type} for {torch_model}")
            if "class_mapping" not in nn.user_metadata:
                raise KrakenInvalidModelException(f"Segmentation model {torch_model} does not contain valid class mapping")
        if not torch_model:
            torch_model = TorchVGSLModel.load_model(str(files(blla.__name__).joinpath("blla.mlmodel")))  # default model
        elif len(torch_model) == 1:
            torch_model = torch_model[0]

    # Segment images
    with util.PROGRESS as progressbar:
        task = progressbar.add_task("Processing...", total=len(images), filename="")
        for fp in images:
            progressbar.update(task, filename=Path(*fp.parts[-min(len(fp.parts), 4):]))
            im = Image.open(fp)
            custom_attributes = {}
            if CUSTOM_KRAKEN:
                custom_attributes["fallback_polygon"] = fallback_polygon
            
            res = blla.segment(
                im=im, 
                text_direction=util.TEXT_DIRECTION_MAPPING[text_direction], 
                model=torch_model, 
                device=device, 
                **custom_attributes
            )
            
            fn = fp.name.split('.')[0] + suffix
            outf = output.joinpath(fn) if output is not None else fp.parent.joinpath(fn)
            log.info(f"Building PageXML {outf.as_posix()}")
            
            xml = segmentation_to_page(
                res, 
                image_width=im.size[0], 
                image_height=im.size[1], 
                creator=creator,
                suppress_lines=suppress_lines, 
                suppress_regions=suppress_regions
            )
            xml.to_file(outf)
            
            if heatmap and CUSTOM_KRAKEN:
                heatmap_data = np.mean(res.heatmap, axis=0)
                heatmap_data = (heatmap_data - heatmap_data.min()) / (heatmap_data.max() - heatmap_data.min()) * 255
                heatmap_data = heatmap_data.astype(np.uint8)
                colormap = np.zeros((256, 3), dtype=np.uint8)
                for i in range(256):
                    colormap[i] = (i, 0, 255 - i)
                heatmap_img = Image.fromarray(colormap[heatmap_data])
                heatmap_fn = fp.name.split('.')[0] + heatmap
                heatmap_img.save(output.joinpath(heatmap_fn) if output is not None else fp.parent.joinpath(heatmap_fn))
                
            progressbar.advance(task)
        progressbar.update(task, filename="Done")
