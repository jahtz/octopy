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

from glob import glob
from typing import Optional, Union
from pathlib import Path

import click

# Callbacks
def callback_paths(ctx, param, value) -> list[Path]:
    if not value:
        raise click.BadParameter("No paths provided", param=param)
    paths = []
    for pattern in value:
        expanded = glob(pattern, recursive=True)
        if not expanded:
            p = Path(pattern)
            if p.exists() and p.is_file():
                paths.append(p)
        else:
            paths.extend(Path(p) for p in expanded if Path(p).is_file())
    if not paths:
        raise click.BadParameter("None of the provided paths or patterns matched existing files.", param=param)
    return paths


def callback_paths_optional(ctx, param, value) -> list[Path]:
    paths = []
    for pattern in value:
        expanded = glob(pattern, recursive=True)
        if not expanded:
            p = Path(pattern)
            if p.exists() and p.is_file():
                paths.append(p)
        else:
            paths.extend(Path(p) for p in expanded if Path(p).is_file())
    return paths


def callback_validate(ctx, param, value) -> Optional[list[str]]:
    """ Parse a baseline/region valid selection pattern to a list of valid baseline/region strings. """
    return None if not value else value.split(',')


def callback_merge(ctx, param, value) -> Optional[dict[str, str]]:
    """ Maps a baseline/region merging pattern to a dict of merge rules. """
    if not value:
        return None
    rules: dict[str, str] = {}
    for rule in value:
        source, target = rule
        if source in rules:
            raise click.BadOptionUsage(param, f"Invalid format: '{source}' cannot be declared multiple times as a source")
        rules[source] = target
    return rules


def expand_paths(paths: Union[Path, list[Path]], glob: str = '*') -> list[Path]:
    """ Expands a list of paths by unpacking directories. """
    result = []
    if isinstance(paths, list):
        for path in paths:
            if path.is_dir():
                result.extend([p for p in path.glob(glob) if p.is_file()])
            else:
                result.append(path)
    elif isinstance(paths, Path):
        if paths.is_dir():
            result.extend([p for p in paths.glob(glob) if p.is_file()])
        else:
            result.append(paths)
    return sorted(result)
