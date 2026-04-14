from __future__ import annotations

from pathlib import Path

from PIL import Image
from torch import nn
from kraken.configs import Config, SegmentationInferenceConfig
from kraken.models import BaseModel, SegmentationBaseModel

from kraken.models.loaders import load_models
from kraken.containers import Segmentation


class OctopySegmentationModel(nn.Module, SegmentationBaseModel):
    _kraken_min_version = '7.0.0'
    model_type: list[str] = ['segmentation']
    
    def __init__(self, weights: Path | str, bbox_pad: int = 0) -> None:
        super().__init__()
        self.weights: Path = Path(weights)
        self.bbox_pad: int = bbox_pad
        
        models: list[BaseModel] = load_models(self.weights, tasks=('segmentation',))  # uses kraken.loaders entry points
        seg_models: list[BaseModel] = [m for m in models if 'segmentation' in getattr(m, 'model_type', [])]
        if not seg_models:
            raise ValueError(f'No segmentation models in {self.weights}')
        self._base: BaseModel = seg_models[0]
        
    def prepare_for_inference(self, config: Config):
        if hasattr(self._base, 'prepare_for_inference'):
            self._base.prepare_for_inference(config)
            
    def predict(self, im: Image.Image, config: SegmentationInferenceConfig | None = None) -> Segmentation:
        pass
        