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
from typing import Optional, Literal

import click
from octopy import segment

from . import util


TEXT_DIRECTION = Literal["hlr", "hrl", "vlr", "vrl"]
log = logging.getLogger("octopy")
log_kraken = logging.getLogger("kraken")
logging.getLogger("PIL").setLevel(logging.CRITICAL)


@click.command("segment")
@click.help_option("--help", hidden=True)
@click.argument(
     "images",
     type=click.Path(exists=False, dir_okay=False, file_okay=True, resolve_path=True),
     callback=util.callback_paths, 
     required=True, 
     nargs=-1
)
@click.option(
     "-m", "--model", "models",
     help="Path to custom segmentation model(s). If not provided, the default Kraken model is used.",
     type=click.Path(exists=True, dir_okay=False, file_okay=True, resolve_path=True, path_type=Path),
     multiple=True
)
@click.option(
     "-o", "--output", "output",
     help="Output directory for processed files. Defaults to the parent directory of each input file.",
     type=click.Path(exists=False, dir_okay=True, file_okay=False, resolve_path=True, path_type=Path)
)
@click.option(
     "-s", "--suffix", "suffix",
     help="Suffix for output PageXML files. Should end with '.xml'.",
     type=click.STRING, 
     default=".xml", 
     show_default=True
)
@click.option(
     "-d", "--device", "device",
     help="Specify the processing device (e.g. 'cpu', 'cuda:0',...). See PyTorch documentation.",
     type=click.STRING, 
     default="cpu", 
     show_default=True
)
@click.option(
     "--creator", "creator",
     help="Metadata: Creator of the PageXML files.",
     type=click.STRING, 
     default="octopy", 
     show_default=True
)
@click.option(
     "--direction", "text_direction",
     help="Text direction of input images.",
     type=click.Choice(["hlr", "hrl", "vlr", "vrl"]),
     default="hlr", 
     show_default=True
)
@click.option(
     "--suppress-lines", "suppress_lines",
     help="Suppress lines in the output PageXML.",
     type=click.BOOL, 
     is_flag=True
)
@click.option(
     "--suppress-regions", "suppress_regions",
     help="Suppress regions in the output PageXML. Creates a single dummy region for the whole image.",
     type=click.BOOL, 
     is_flag=True
)
@click.option(
     "--fallback", "fallback_polygon",
     help="Use a default bounding box when the polygonizer fails to create a polygon around a baseline (in pixels). ",
     type=click.INT
)
@click.option(
     "--heatmap", "heatmap",
     help="Generate a heatmap image alongside the PageXML output. "
          "Specify the file extension for the heatmap (e.g., `.hm.png`).",
     type=click.STRING
)
@click.option(
     "--logging", "logging_level",
     help="Set logging level.", 
     type=click.Choice(["ERROR", "WARNING", "INFO", "DEBUG"]),
     default="ERROR",
     show_default=True
)
def segment_cli(
     images: list[Path],
     models: list[Path],
     output: Optional[Path] = None,
     suffix: str = ".xml",
     device: str = "cpu",
     creator: str = "octopy",
     text_direction: TEXT_DIRECTION = "hlr",
     suppress_lines: bool = False,
     suppress_regions: bool = False,
     fallback_polygon: Optional[int] = None,
     heatmap: Optional[str] = None,
     logging_level: Literal["ERROR", "WARNING", "INFO", "DEBUG"] = "ERROR"
) -> None:
     """
     Segment images using Kraken.

     IMAGES: Specify one or more image files to segment.
     Supports multiple file paths, wildcards, or directories (with the -g option).
     """
     log.setLevel(logging_level)
     log_kraken.setLevel(logging_level)
     if len(images) < 1:
          click.BadArgumentUsage("No images found")
          return
     if output is not None:
          output.mkdir(parents=True, exist_ok=True)
     
     segment(
          images=images, 
          models=list(models), 
          output=output, 
          suffix=suffix if suffix.startswith('.') else '.' + suffix,
          device=device, 
          creator=creator,
          suppress_lines=suppress_lines, 
          suppress_regions=suppress_regions, 
          text_direction=text_direction,
          fallback_polygon=fallback_polygon, 
          heatmap=None if heatmap is None else heatmap if heatmap.startswith('.') else '.' + heatmap
     )
