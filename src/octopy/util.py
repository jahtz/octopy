from typing import Union

import rich_click as click


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
