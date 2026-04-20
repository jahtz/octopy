# SPDX-License-Identifier: Apache-2.0
import logging
from pathlib import PosixPath
import warnings

from PIL import Image
import torch

from .segment import Segmenter
from .train import Trainer, training_data_config, training_model_config


__all__: list[str] = ['Segmenter', 'Trainer', 'training_data_config', 'training_model_config']
logger: logging.Logger = logging.getLogger('octopy')


for name in ('kraken', 'lightning', 'lightning.pytorch', 'lightning.fabric'):
    lg: logging.Logger = logging.getLogger(name)
    lg.handlers.clear()
    lg.propagate = True
    lg.setLevel(logger.level)

warnings.filterwarnings('ignore', message=r'You called `self\.log\(.*\)` but have no logger configured\.')
torch.serialization.add_safe_globals([PosixPath])
Image.MAX_IMAGE_PIXELS: int = 20000 ** 2
