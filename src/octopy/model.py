import json
import logging
from pathlib import Path
from typing import Any


logger: logging.Logger = logging.getLogger(__name__)


def inspect_model(model: Path) -> dict[str, Any]:
    metadata: dict = {}
    try:
        if model.suffix.lower() == '.safetensors':
            from safetensors import safe_open
            with safe_open(str(model), framework='pt', device='cpu') as f:
                raw_meta = f.metadata() or {}
                for k, v in raw_meta.items():
                    try:
                        metadata[k] = json.loads(v)
                    except Exception:
                        metadata[k] = v
        else:
            from kraken.lib.vgsl import TorchVGSLModel
            nn: TorchVGSLModel = TorchVGSLModel.load_model(model)
            metadata = dict(nn.user_metadata or {})
    except Exception as exc:
        logger.error(f'Could not load model: {exc}')
        raise
    
    metadata.pop('accuracy', None)
    return metadata
