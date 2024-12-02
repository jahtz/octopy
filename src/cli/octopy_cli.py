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

import rich_click as click

from .shrink_cli import shrink_cli
from .segment_cli import segment_cli
from .segtrain_cli import segtrain_cli


__version__ = "5.2.9"
__prog__ = "octopy"

click.rich_click.SHOW_ARGUMENTS = True
click.rich_click.MAX_WIDTH = 96
click.rich_click.RANGE_STRING = ""
click.rich_click.OPTION_GROUPS = {
    "octopy segment": [
        {
            "name": "Input",
            "options": ["images"],
        },
        {
            "name:": "Options",
            "options": ["--glob", "--model", "--output", "--suffix", "--device"]
        },
        {
            "name": "Fine-Tuning",
            "options": ["--creator", "--direction", "--suppress-lines", "--suppress-regions", "--fallback", "--heatmap"]
        },
    ],
    "octopy segtrain": [
        {
            "name": "Input",
            "options": ["--gt", "--gt-glob", "--eval", "--eval-glob", "--partition", "--model"],
        },
        {
            "name:": "Options",
            "options": ["--output", "--name", "--device", "--workers", "--threads", "--resize", "--suppress-regions",
                        "--suppress-baselines", "--valid-regions", "--valid-baselines", "--merge-regions",
                        "--merge-baselines", "--verbose"]
        },
        {
            "name": "Hyperparameters",
            "options": ["--line-width", "--line-width", "--padding", "--freq", "--quit", "--epochs", "--min-epochs",
                        "--lag", "--optimizer", "--lrate", "--momentum", "--weight-decay", "--schedule",
                        "--completed-epochs", "--augment", "--step-size", "--gamma", "--rop-factor", "--rop-patience",
                        "--cos-t-max", "--cos-min-lr", "--warmup"]
        }
    ],
    "octopy shrink": [],
}


@click.group()
@click.help_option("--help")
@click.version_option(__version__,
                      "--version",
                      prog_name=__prog__,
                      message=f"{__prog__} v{__version__} - Developed at Centre for Philology and Digitality (ZPD), "
                              f"University of Würzburg")
def cli(**kwargs):
    """
    Command line tool layout analysis and OCR of historical prints using Kraken.
    """

cli.add_command(segment_cli)
cli.add_command(segtrain_cli)
cli.add_command(shrink_cli)
