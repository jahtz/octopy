from pypxml import PageType

TEXT_DIRECTION_MAPPING = {
    "hlr": "horizontal-lr",
    "hrl": "horizontal-rl",
    "vlr": "vertical-lr",
    "vrl": "vertical-rl",
}

SEGMENTATION_MAPPING = {
    # Region classes
    "maths": (PageType.MathsRegion, None),
    "graphic": (PageType.GraphicRegion, None),
    "image": (PageType.ImageRegion, None),
    "separator": (PageType.SeparatorRegion, None),
    "table": (PageType.TableRegion, None),
    "music": (PageType.MusicRegion, None),

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

    # Fallback class
    "unknown": (PageType.UnknownRegion, None)
}

