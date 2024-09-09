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

# This project includes code from the kraken project,
# available at https://github.com/mittagessen/kraken and licensed under
# Apache 2.0 lincese https://github.com/mittagessen/kraken/blob/main/LICENSE.

import click

from modules.segtrain import segtrain_cli
from modules.segment import segment_cli


@click.group()
@click.help_option('--help')
@click.version_option('5.2.9', '--version',
                      prog_name='octopy',
                      message='\n%(prog)s v%(version)s - Developed at Centre for Philology and Digitality (ZPD), University of Würzburg')
def octopy_cli(**kwargs):
    """
    Octopy main entry point.

    Wrapper for Kraken by Mittagessen
    """
    pass


# add modules
octopy_cli.add_command(segtrain_cli)
octopy_cli.add_command(segment_cli)


if __name__ == '__main__':
    octopy_cli()
