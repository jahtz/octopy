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
import click
from rich.logging import RichHandler

from .segment_cli import segment_cli
from .segtrain_cli import segtrain_cli


__version__ = "5.2.9"
__prog__ = "octopy"
__footer__ = "Developed at Centre for Philology and Digitality (ZPD), University of Würzburg"

logging.basicConfig(
    level="NOTSET",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(markup=True)]
)
logger = logging.getLogger("octopy")


@click.group(epilog=__footer__)
@click.help_option("--help")
@click.version_option(
    __version__, "--version",
    prog_name=__prog__,
    message=f"{__prog__} v{__version__}\n{__footer__}"
)
def cli(**kwargs):
    """
    Command line tool layout analysis and OCR of historical prints using Kraken.
    """

cli.add_command(segment_cli)
cli.add_command(segtrain_cli)
