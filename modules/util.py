from pathlib import Path
import shlex
import glob

import click


def validate_merging(ctx, param, value) -> dict[str, str] | None:
    """ Maps baseline/region merging to a dict of merge structures. """

    # Forked from Kraken:
    # https://github.com/mittagessen/kraken/blob/main/kraken/ketos/segmentation.py#L42

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


def parse_files(ctx, param, value) -> list[Path]:
    """ Parse ground truth files. """
    files = []
    for path in value:
        files.extend([Path(x) for x in glob.glob(path)])
    return files


def parse_file(ctx, param, value) -> Path:
    """ Parse ground truth file. """
    return None if value is None else Path(value)