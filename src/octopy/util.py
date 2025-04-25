from typing import Union

import rich_click as click
from rich.progress import (Progress, SpinnerColumn, TextColumn, BarColumn, MofNCompleteColumn, TimeElapsedColumn, 
                           TimeRemainingColumn)

from pypxml import PageType


TEXT_DIRECTION_MAPPING = {
    "hlr": "horizontal-lr",
    "hrl": "horizontal-rl",
    "vlr": "vertical-lr",
    "vrl": "vertical-rl",
}

SEGMENTATION_CLASS_MAPPING = {
    # Region classes
    "advert": (PageType.AdvertRegion, None),
    "chart": (PageType.ChartRegion, None),
    "chem": (PageType.ChemRegion, None),
    "custom": (PageType.CustomRegion, None),
    "graphic": (PageType.GraphicRegion, None),
    "image": (PageType.ImageRegion, None),
    "line drawing": (PageType.LineDrawingRegion, None),
    "map": (PageType.MapRegion, None),
    "maths": (PageType.MathsRegion, None),
    "music": (PageType.MusicRegion, None),
    "noise": (PageType.NoiseRegion, None),
    "separator": (PageType.SeparatorRegion, None),
    "table": (PageType.TableRegion, None),
    "unknown": (PageType.UnknownRegion, None),

    # TextRegion classes
    "text": (PageType.TextRegion, "paragraph"),
    "paragraph": (PageType.TextRegion, "paragraph"),
    "endnote": (PageType.TextRegion, "endnote"),
    "header": (PageType.TextRegion, "header"),
    "heading": (PageType.TextRegion, "heading"),
    "signature-mark": (PageType.TextRegion, "signature-mark"),
    "catch-word": (PageType.TextRegion, "catch-word"),
    "drop-capital": (PageType.TextRegion, "drop-capital"),
    "page-number": (PageType.TextRegion, "page-number"),
    "footnote": (PageType.TextRegion, "footnote"),
    "marginalia": (PageType.TextRegion, "marginalia"),
    "caption": (PageType.TextRegion, "caption"),
    "other": (PageType.TextRegion, "other"),

    # Default Kraken (blla.mlmodel) classes
    "Title": (PageType.TextRegion, "caption"),
    "Illustration": (PageType.ImageRegion, None),
    "Commentary": (PageType.TextRegion, "footnote"),
}

progress = Progress(TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                    MofNCompleteColumn(),
                    TextColumn("•"),
                    TimeElapsedColumn(),
                    TextColumn("•"),
                    TimeRemainingColumn(),
                    TextColumn("• {task.fields[filename]}"))

spinner = Progress(SpinnerColumn(), 
                    TextColumn("[progress.description]{task.description}"), 
                    transient=True)


def kraken_to_string(coords: list[Union[list[int], tuple[int, int]]]) -> str:
    """
    Convert a list of kraken points to a PageXML points string.
    Args:
        coords: List of kraken coords.
    Returns:
        A string containing the points in PageXML format.
    """
    return ' '.join([f"{point[0]},{point[1]}" for point in coords])


def device_parser(d: str) -> tuple[str, Union[str, list[int]]]:
    """
    Parses the input device string to a pytorch accelerator and device string.
    Args:
        d: Encoded device string (see PyTorch documentation).
    Returns:
        Tuple containing accelerator string and device integer/string.
    """
    auto_devices = ["cpu", "mps"]
    acc_devices = ["cuda", "tpu", "hpu", "ipu"]
    if d in auto_devices:
        return d, "auto"
    elif any([d.startswith(x) for x in acc_devices]):
        dv, i = d.split(':')
        if dv == "cuda":
            dv = "gpu"
        return dv, [int(i)]
    else:
        raise click.BadParameter(f"Invalid device string: {d}")
