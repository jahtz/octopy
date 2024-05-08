from pathlib import Path
from glob import glob


def normalize_suffix(suffix: str) -> str:
    """ add leading dot to suffix if missing """
    if not suffix.startswith('.'):
        suffix = '.' + suffix
    return suffix


def path_parser(inputs: tuple) -> list[Path]:
    """ Converts a tuple of path strings containing glob patterns to a list of Path objects. """
    files = []
    for e in inputs:
        files.extend([Path(x) for x in glob(e)])
    return files


if __name__ == '__main__':
    print(path_parser(('/home/janik/Desktop/examples/input/0046.png', '/home/janik/Desktop/examples/processing/*.bin.png')))