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
from pathlib import Path
import inspect
from typing import Optional, Union, Literal

import lightning as _  # fixes "Segmentation Fault (Core dumped)"
from importlib_resources import files
import numpy as np
from PIL import Image
from pypxml import PageXML, PageType
from kraken import blla
from kraken.lib.vgsl import TorchVGSLModel
from kraken.containers import Segmentation
from kraken.lib.exceptions import KrakenInvalidModelException

from . import util

TEXT_DIRECTION = Literal["hlr", "hrl", "vlr", "vrl"]
custom_kraken = len(inspect.signature(blla.segment).parameters) > 8  # check if the modified kraken version is installed
logger = logging.getLogger(__name__)


def segmentation_to_page(res: Segmentation,
                         image_width: int,
                         image_height: int,
                         creator: str,
                         suppress_lines: bool = False,
                         suppress_regions: bool = False) -> PageXML:
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
    pagexml = PageXML(image_height, image_width,
                      filename=f"{fn[0]}.{fn[-1]}",
                      creator=creator)
    rc, lc = 1, 1
    if suppress_regions:
        dummy_region = pagexml.create_element(PageType.TextRegion, type="paragraph", id="rdummy")
        dummy_coords = [(0, 0), (image_width, 0), (image_width, image_height), (0, image_height)]
        dummy_region.create_element(PageType.Coords, points=util.kraken_to_string(dummy_coords))
        if suppress_lines:
            return pagexml
        for line in res.lines:
            textline = dummy_region.create_element(PageType.TextLine, id=f"l{lc}")
            if boundary := line.boundary:
                textline.create_element(PageType.Coords, points=util.kraken_to_string(boundary))
            if baseline := line.baseline:
                textline.create_element(PageType.Baseline, points=util.kraken_to_string(baseline))
            lc += 1
    else:
        for region_class, found_regions in res.regions.items():
            if region_class not in util.SEGMENTATION_CLASS_MAPPING:
                logger.warning(f"Unknown region class: {region_class}")
                continue
            pagetype, type_attribute = util.SEGMENTATION_CLASS_MAPPING[region_class]
            for found_region in found_regions:
                region = pagexml.create_element(pagetype, type=type_attribute, id=f"r{rc}")
                region.create_element(PageType.Coords, points=util.kraken_to_string(found_region.boundary))
                if suppress_lines:
                    continue
                for found_line in res.lines:
                    if found_region.id in found_line.regions:
                        textline = region.create_element(PageType.TextLine, id=f"l{lc}")
                        if boundary := found_line.boundary:
                            textline.create_element(PageType.Coords, points=util.kraken_to_string(boundary))
                        if baseline := found_line.baseline:
                            textline.create_element(PageType.Baseline, points=util.kraken_to_string(baseline))
                        lc += 1
                rc += 1
    return pagexml


def segment(images: Union[Path, list[Path]],
            models: Optional[Union[Path, list[Path]]] = None,
            output: Optional[Path] = None,
            output_suffix: str = ".xml",
            device: str = "cpu",
            creator: str = "octopy",
            text_direction: TEXT_DIRECTION = "hlr",
            suppress_lines: bool = False,
            suppress_regions: bool = False,
            fallback_polygon: Optional[int] = None,
            heatmap: Optional[str] = None):
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
    if not custom_kraken:
        logger.warning("Some features are not available due to the installed Kraken version")
        
    if not isinstance(images, list):
        images = [images]
    if models is not None and not isinstance(models, list):
        models = [models]

    # Load models
    torch_model = []
    with util.spinner as spinner:
        spinner.add_task(description="Loading models", total=None)
        for model in models:
            try:
                nn = TorchVGSLModel.load_model(model)
                torch_model.append(nn)
            except Exception as e:
                logger.error(f"Could not load model ({model}): {e}")
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
    with util.progress as p:
        task = p.add_task("Processing images", total=len(images), filename="")
        for fp in images:
            p.update(task, filename=Path(*fp.parts[-min(len(fp.parts), 4):]))
            im = Image.open(fp)
            custom_attributes = {}
            if custom_kraken:
                custom_attributes["fallback_polygon"] = fallback_polygon
            res = blla.segment(im=im, text_direction=util.TEXT_DIRECTION_MAPPING[text_direction], model=torch_model, 
                                device=device, **custom_attributes)
            outname = fp.name.split('.')[0] + output_suffix
            outfile = output.joinpath(outname) if output is not None else fp.parent.joinpath(outname)
            logger.info(f"Building PageXML {outfile.as_posix()}")
            xml = segmentation_to_page(res, image_width=im.size[0], image_height=im.size[1], creator=creator,
                                       suppress_lines=suppress_lines, suppress_regions=suppress_regions)
            xml.to_file(outfile)
            if heatmap and custom_kraken:
                heatmap_data = np.mean(res.heatmap, axis=0)
                heatmap_data = (heatmap_data - heatmap_data.min()) / (heatmap_data.max() - heatmap_data.min()) * 255
                heatmap_data = heatmap_data.astype(np.uint8)
                colormap = np.zeros((256, 3), dtype=np.uint8)
                for i in range(256):
                    colormap[i] = (i, 0, 255 - i)
                heatmap_img = Image.fromarray(colormap[heatmap_data])
                heatmap_name = fp.name.split('.')[0] + heatmap
                heatmap_img.save(output.joinpath(heatmap_name) if output is not None else fp.parent.joinpath(heatmap_name))
            p.update(task, advance=1)
        p.update(task, filename="Done")
