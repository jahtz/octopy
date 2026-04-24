# SPDX-License-Identifier: Apache-2.0
"""
This plugin replaces Kraken's default polygonizer behavior entirely
"""
from __future__ import annotations

import logging


logger: logging.Logger = logging.getLogger('octopy')


class OctopySegmenter:
    
    @staticmethod
    def register() -> None:
        raise NotImplementedError('This is currently work in progress')
        logger.info('Plugin: OctopySegmenter registered')
