from typing import Optional, Union
from pathlib import Path

import rich_click as click


# Callbacks
def paths_callback(ctx, param, value: list[str]) -> list[Path]:
    """ Parse a list of click paths to a list of pathlib Path objects. """
    return [] if value is None else list([Path(p) for p in value])

def path_callback(ctx, param, value: str) -> Optional[Path]:
    """ Parse a click path to a pathlib Path object. """
    return None if value is None else Path(value)

def suffix_callback(ctx, param, value: str) -> str:
    """ Parses a string to a valid suffix. """
    return value if value.startswith('.') else f".{value}"#

def validate_callback(ctx, param, value) -> Optional[list[str]]:
    """ Parse a baseline/region valid selection pattern to a list of valid baseline/region strings. """
    return None if not value else value.split(',')

def merge_callback(ctx, param, value) -> Optional[dict[str, str]]:
    """ Maps a baseline/region merging pattern to a dict of merge rules. """
    if not value:
        return None
    merge_rules: dict[str, str] = {}
    for rule in value:
        tokens: list[str] = rule.split(':')
        if len(tokens) != 2:
            raise click.BadParameter(f"Invalid merging rule: {rule}. "
                                     f"Mappings must be in format src:target or src1,src2:target")
        from_side: list[str] = tokens[0].split(',')
        to_side: str = tokens[1]
        for key in from_side:
            merge_rules[key.strip()] = to_side.strip()
    return merge_rules

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