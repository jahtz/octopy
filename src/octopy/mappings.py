# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from pypxml import PageType


# https://ocr-d.de/de/gt-guidelines/pagexml/pagecontent_xsd_Simple_Type_pc_ReadingDirectionSimpleType.html#ReadingDirectionSimpleType
default_direction_mapping: dict[str, str] = {
    'horizontal-lr': 'left-to-right',
    'horizontal-rl': 'right-to-left',
    'vertical-lr': 'bottom-to-top',
    'vertical-rl': 'top-to-bottom'
}

default_region_mapping: dict[str, tuple[PageType, str | None]] = {
    # Region classes
    'advert': (PageType.AdvertRegion, None),
    'chart': (PageType.ChartRegion, None),
    'chem': (PageType.ChemRegion, None),
    'custom': (PageType.CustomRegion, None),
    'graphic': (PageType.GraphicRegion, None),
    'image': (PageType.ImageRegion, None),
    'line drawing': (PageType.LineDrawingRegion, None),
    'map': (PageType.MapRegion, None),
    'maths': (PageType.MathsRegion, None),
    'music': (PageType.MusicRegion, None),
    'noise': (PageType.NoiseRegion, None),
    'separator': (PageType.SeparatorRegion, None),
    'table': (PageType.TableRegion, None),
    'unknown': (PageType.UnknownRegion, None),

    # TextRegion classes
    'paragraph': (PageType.TextRegion, 'paragraph'),
    'endnote': (PageType.TextRegion, 'endnote'),
    'header': (PageType.TextRegion, 'header'),
    'heading': (PageType.TextRegion, 'heading'),
    'signature-mark': (PageType.TextRegion, 'signature-mark'),
    'catch-word': (PageType.TextRegion, 'catch-word'),
    'drop-capital': (PageType.TextRegion, 'drop-capital'),
    'page-number': (PageType.TextRegion, 'page-number'),
    'footnote': (PageType.TextRegion, 'footnote'),
    'marginalia': (PageType.TextRegion, 'marginalia'),
    'caption': (PageType.TextRegion, 'caption'),
    'other': (PageType.TextRegion, 'other'),

    # Default Kraken (blla.mlmodel) classes
    'text': (PageType.TextRegion, 'paragraph'),
}
