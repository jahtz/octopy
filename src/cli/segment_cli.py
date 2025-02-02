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
from typing import Optional, Literal

import rich_click as click
from rich import print as rprint

from octopy import segment
from .util import paths_callback, path_callback, suffix_callback, expand_paths


TEXT_DIRECTION = Literal["hlr", "hrl", "vlr", "vrl"]


@click.command("segment")
@click.help_option("--help", hidden=True)
@click.argument("images",
                type=click.Path(exists=True, dir_okay=True, file_okay=True, resolve_path=True),
                callback=paths_callback, required=True, nargs=-1)
@click.option("-g", "--glob", "glob",
              help="Glob pattern for matching images in directories. (used with directories in IMAGES).",
              type=click.STRING, default="*.ocropus.bin.png", required=False, show_default=True)
@click.option("-m", "--model", "models",
              help="Path to custom segmentation model(s). If not provided, the default Kraken model is used.",
              type=click.Path(exists=True, dir_okay=False, file_okay=True, resolve_path=True),
              callback=paths_callback, required=False, multiple=True)
@click.option("-o", "--output", "output",
              help="Output directory for processed files. Defaults to the parent directory of each input file.",
              type=click.Path(exists=False, dir_okay=True, file_okay=False, resolve_path=True),
              callback=path_callback, required=False)
@click.option("-s", "--suffix", "output_suffix",
              help="Suffix for output PageXML files. Should end with '.xml'.",
              type=click.STRING, callback=suffix_callback, required=False, default=".xml", show_default=True)
@click.option("-d", "--device", "device",
              help="Specify the processing device (e.g. 'cpu', 'cuda:0',...). "
                   "Refer to PyTorch documentation for supported devices.",
              type=click.STRING, required=False, default="cpu", show_default=True)
@click.option("--creator", "creator",
              help="Metadata: Creator of the PageXML files.",
              type=click.STRING, required=False, default="octopy", show_default=True)
@click.option("--direction", "text_direction",
              help="Text direction of input images.",
              type=click.Choice(["hlr", "hrl", "vlr", "vrl"]),
              required=False, default="hlr", show_default=True)
@click.option("--suppress-lines", "suppress_lines",
                help="Suppress lines in the output PageXML.",
                type=click.BOOL, is_flag=True, required=False)
@click.option("--suppress-regions", "suppress_regions",
                help="Suppress regions in the output PageXML. Creates a single dummy region for the whole image.",
                type=click.BOOL, is_flag=True, required=False)
@click.option("--fallback", "fallback_polygon",
              help="Use a default bounding box when the polygonizer fails to create a polygon around a baseline. "
                   "Requires a box height in pixels.",
              type=click.INT, required=False)
@click.option("--heatmap", "heatmap",
              help="Generate a heatmap image alongside the PageXML output. "
                   "Specify the file extension for the heatmap (e.g., `.hm.png`).",
              type=click.STRING, callback=suffix_callback, required=False)
def segment_cli(images: list[Path],
                models: list[Path],
                glob: str = "*.ocropus.bin.png",
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

    IMAGES: Specify one or more image files to segment.
    Supports multiple file paths, wildcards, or directories (with the -g option).
    """
    images = expand_paths(images, glob)
    if len(images) < 1:
        rprint("[red bold]Error:[/red bold] No images to segment")
        return
    rprint(f"Segmenting {len(images)} images")
    if output is not None:
        output.mkdir(parents=True, exist_ok=True)
    segment(images=images, models=models, output=output, output_suffix=output_suffix, device=device, creator=creator,
            suppress_lines=suppress_lines, suppress_regions=suppress_regions, text_direction=text_direction,
            fallback_polygon=fallback_polygon, heatmap=heatmap)
