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
from typing import Optional

import rich_click as click
from rich import print as rprint
from rich.progress import track

from octopy import region_shrink
from .util import paths_callback, path_callback, suffix_callback, expand_paths


@click.command("shrink")
@click.help_option("--help", hidden=True)
@click.argument("pagexml",
                type=click.Path(exists=True, dir_okay=True, file_okay=True, resolve_path=True),
                callback=paths_callback, required=True, nargs=-1)
@click.option("-g", "--glob", "glob",
              help="Glob pattern for matching PageXML files in directories. (used with directories in PAGEXML).",
              type=click.STRING, default="*.xml", required=False, show_default=True)
@click.option("-o", "--output", "output",
              help="Output directory for processed files. Defaults to the parent directory of each input file.",
              type=click.Path(exists=False, dir_okay=True, file_okay=False, resolve_path=True),
              callback=path_callback, required=False)
@click.option("-i", "--input-suffix", "input_suffix",
              help="Suffix for image selection. Should match full suffix of input PageXML files.",
              type=click.STRING, callback=suffix_callback, required=False, default=".bin.png", show_default=True)
@click.option("-s", "--output-suffix", "output_suffix",
              help="Suffix for shrunken PageXML files. Should end with '.xml'. Could overwrite input files.",
              type=click.STRING, callback=suffix_callback, required=False, default=".xml", show_default=True)
@click.option("-p", "--padding", "padding",
              help="Padding around the shrunken regions in pixels.",
              type=click.INT, required=False, default=5, show_default=True)
@click.option("-h", "--horizontal", "h_smoothing",
              help="The higher, the more horizontal smoothing is applied.",
              type=click.INT, required=False, default=3, show_default=True)
@click.option("-v", "--vertical", "v_smoothing",
              help="The higher, the more vertical smoothing is applied.",
              type=click.INT, required=False, default=3, show_default=True)
@click.option("-vr", "--valid-region", "valid_regions",
              help="Valid regions for shrinking. If nothing is provided, all regions are shrunk. "
                   "Multiple selections are possible.",
              type=click.STRING, required=False, multiple=True)
def shrink_cli(pagexml: list[Path], glob: str = "*.xml", output: Optional[Path] = None, input_suffix: str = ".bin.png",
               output_suffix: str = ".xml", padding: int = 5, h_smoothing: int = 3, v_smoothing: int = 3,
               valid_regions: Optional[list[str]] = None) -> None:
    """
    Shrink region polygons of PageXML files.

    PAGEXML: Specify one or more PageXML files to shrink.
    Supports multiple file paths, wildcards, or directories (with the -g option).
    """
    pagexml = expand_paths(pagexml, glob)
    for i in track(range(len(pagexml)), description="Shrinking regions..."):
        pxml = pagexml[i]
        image = pxml.parent.joinpath(pxml.name.split('.')[0] + input_suffix)
        if not image.exists():
            rprint(f"[red]Image file {image} not found![/red]")
            continue
        if output is None:
            outfile = pxml.parent.joinpath(pxml.name.split('.')[0] + output_suffix)
        else:
            outfile = output.joinpath(pxml.name.split('.')[0] + output_suffix)
        region_shrink(pagexml=pxml,
                      image=image,
                      padding=padding,
                      h_smoothing=h_smoothing,
                      v_smoothing=v_smoothing,
                      valid_regions=None if not valid_regions else valid_regions).to_xml(outfile)
