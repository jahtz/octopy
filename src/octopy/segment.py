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

from pathlib import Path
from typing import Optional, Union, Literal

import lightning as _  # fixes "Segmentation Fault (Core dumped)"
from importlib_resources import files
from rich import print as rprint
from rich.progress import track, Progress, SpinnerColumn, TextColumn
from PIL import Image
from pypxml import PageXML, PageType
from kraken import blla
from kraken.lib.vgsl import TorchVGSLModel
from kraken.containers import Segmentation
from kraken.lib.exceptions import KrakenInvalidModelException

from .util import is_bitonal, kraken_to_string
from .mappings import TEXT_DIRECTION_MAPPING, SEGMENTATION_MAPPING


TEXT_DIRECTION = Literal["hlr", "hrl", "vlr", "vrl"]


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
    pxml = PageXML.new(creator=creator,
                       imageFilename=f"{fn[0]}.{fn[-1]}",
                       imageWidth=str(image_width),
                       imageHeight=str(image_height))
    rc, lc = 1, 1
    if suppress_regions:
        relement = pxml.create_element(PageType.TextRegion, type="paragraph", id=f"r_dummy")
        coords = [(0, 0), (image_width, 0), (image_width, image_height), (0, image_height)]
        relement.create_element(PageType.Coords, points=kraken_to_string(coords))
        if not suppress_lines:
            for line in res.lines:
                lelement = relement.create_element(PageType.TextLine, id=f"l_{lc:04d}")
                lc += 1
                if line.boundary:
                    lelement.create_element(PageType.Coords, points=kraken_to_string(line.boundary))
                if line.baseline:
                    lelement.create_element(PageType.Baseline, points=kraken_to_string(line.baseline))
    else:
        for region_type, regions in res.regions.items():
            if region_type not in SEGMENTATION_MAPPING:
                rprint(f"[orange bold]WARNING:[/orange bold] Unknown region class {region_type}")
                continue
            xmltype, rtype = SEGMENTATION_MAPPING[region_type]
            for region in regions:
                relement = pxml.create_element(xmltype, type=rtype, id=f"r_{rc:04d}")
                relement.create_element(PageType.Coords, points=kraken_to_string(region.boundary))
                rc += 1
                if not suppress_lines:
                    for line in res.lines:
                        if region.id in line.regions:
                            lelement = relement.create_element(PageType.TextLine, id=f"l_{lc:04d}")
                            lc += 1
                            if line.boundary:
                                lelement.create_element(PageType.Coords, points=kraken_to_string(line.boundary))
                            if line.baseline:
                                lelement.create_element(PageType.Baseline, points=kraken_to_string(line.baseline))
    return pxml


def segment(images: Union[Path, list[Path]],
            models: Optional[Union[Path, list[Path]]] = None,
            output: Optional[Path] = None,
            output_suffix: str = ".xml",
            device: str = "cpu",
            creator: str = "octopy",
            text_direction: TEXT_DIRECTION = "hlr",
            suppress_lines: bool = False,
            suppress_regions: bool = False,
            line_fallback: bool = False,
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
        line_fallback: Use a default bounding box when the polygonizer fails to create a polygon around a baseline.
        heatmap: Generate a heatmap image alongside the PageXML output.
            Specify the file extension for the heatmap (e.g., `.hm.png`).
    """
    # Load models
    torch_model = []
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), transient=True) as spinner:
        spinner.add_task(description="Loading models...", total=None)
        for model in models:
            try:
                nn = TorchVGSLModel.load_model(model)
                torch_model.append(nn)
                print(nn.user_metadata)
            except Exception as e:
                rprint(f"[red bold]Error:[/red bold] Could not load model.\n{e}")
                continue
            if nn.model_type != 'segmentation':
                raise KrakenInvalidModelException(f'Invalid model type {nn.model_type} for {torch_model}')
            if 'class_mapping' not in nn.user_metadata:
                raise KrakenInvalidModelException(f'Segmentation model {torch_model} does not contain valid class mapping')
        if not torch_model:
            torch_model = TorchVGSLModel.load_model(str(files(blla.__name__).joinpath('blla.mlmodel')))  # default model
        elif len(torch_model) == 1:
            torch_model = torch_model[0]

    # Segment images
    for i in track(range(len(images)), description="Segmenting images..."):
        im = Image.open(images[i])

        polygon = False if not line_fallback else is_bitonal(im)
        if line_fallback and not polygon:
            rprint(f"[orange bold]WARNING:[/orange bold] No fallback polygon available. "
                   f"--line-fallback requires a binary input image.")

        res = blla.segment(im=im, text_direction=TEXT_DIRECTION_MAPPING[text_direction], model=torch_model,
                           device=device)  #  TODO: update Kraken to calculate default polygons
        outname = images[i].name.split('.')[0] + output_suffix
        outfile = output.joinpath(outname) if output is not None else images[0].parent.joinpath(outname)
        xml = segmentation_to_page(res, image_width=im.size[0], image_height=im.size[1], creator=creator,
                                   suppress_lines=suppress_lines, suppress_regions=suppress_regions)
        xml.to_xml(outfile)
        if heatmap:
            pass
            # TODO: Implement heatmap generation
