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
# Apache 2.0 license https://github.com/mittagessen/kraken/blob/main/LICENSE.

from pathlib import Path
from typing import Optional
import shlex

import click


##### CLICK CALLBACKS ##### 
def parse_path_list(ctx, param, value) -> list[Path]:
    """ Parse a list of click paths to a list of pathlib Path objects. """
    return None if value is None else list([Path(x) for x in value])


def parse_path(ctx, param, value) -> Path:
    """ Parse a click path to a pathlib Path object. """
    return None if value is None else Path(value)


def parse_suffix(ctx, param, value) -> Optional[str]:
    """ Parses a string to a valid suffix. """
    if value is None:
        return None
    return value if value.startswith('.') else f'.{value}'


# Forked from Kraken:
# https://github.com/mittagessen/kraken/blob/main/kraken/ketos/segmentation.py#L42
def validate_merging(ctx, param, value) -> Optional[dict[str, str]]:
    """ Maps baseline/region merging to a dict of merge structures. """
    if not value:
        return None
    merge_dict: dict[str, str] = {}
    try:
        for m in value:
            lexer = shlex.shlex(m, posix=True)
            lexer.wordchars += r'\/.+-()=^&;,.'
            tokens = list(lexer)
            if len(tokens) != 3:
                raise ValueError
            k, _, v = tokens
            merge_dict[v] = k  # type: ignore
    except Exception:
        raise click.BadParameter('Mappings must be in format target:src')
    return merge_dict


##### PATH OPERATIONS #####
def expand_path_list(paths: list[Path], glob: str = '*') -> list[Path]:
    """ Expands a list of paths by unpacking directories. """
    path_list = []
    for path in paths:
        if path.is_dir():
            path_list.extend(sorted([fp for fp in path.glob(glob) if fp.is_file()]))
        else:
            path_list.append(path)
    return path_list


def expand_path(path: Path, glob: str = '*') -> list[Path]:
    """ Expand a path with a glob expression. """
    if not path.is_dir():
        return [path]
    return list(sorted([fp for fp in path.glob(glob) if fp.is_file()]))
