# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from dataclasses import dataclass
import logging
from os import PathLike
from pathlib import Path
from shutil import copy
from typing import Literal

from kraken.lib.train import KrakenTrainer, SegmentationModel
from kraken.lib.default_specs import SEGMENTATION_HYPER_PARAMS
from lightning.pytorch import seed_everything
from rich.console import Console
from rich.table import Table
from threadpoolctl import threadpool_limits


logger: logging.Logger = logging.getLogger(__name__)


@dataclass
class DataConfig:
    output: PathLike | str
    train_data: list[PathLike | str]
    eval_data: list[PathLike | str] | None = None
    partition: float = 0.9
    model_name: str = 'model'
    image_extension: str | None = None
    model: PathLike | str | None = None
    deterministic: bool = False 
    position: Literal['baseline', 'centerline', 'topline'] = 'baseline'
    suppress_regions: bool = False
    suppress_baselines: bool = False
    valid_regions: list[str] | None = None
    valid_baselines: list[str] | None = None
    merge_regions: dict[str, list[str]] | None = None
    merge_baselines: dict[str, list[str]] | None = None


@dataclass
class TrainerConfig:
    vgsl: str = '[1,1800,0,3 Cr7,7,64,2,2 Gn32 Cr3,3,128,2,2 Gn32 Cr3,3,128 Gn32 Cr3,3,256 Gn32 Cr3,3,256 Gn32 Lbx32 Lby32 Cr1,1,32 Gn32 Lby32 Lbx32]'
    resize: Literal['union', 'new', 'fail'] = 'new'
    frequency: float = 1.0
    line_width: int =  8
    padding: tuple[int, int] = (0, 0)
    quit: Literal['fixed', 'early'] = 'fixed'
    epochs: int = 50
    min_epochs: int = 0
    lag: int = 10
    optimizer: Literal['Adam', 'AdamW', 'SGD', 'RMSprop', 'Lamb'] = 'AdamW'
    lrate: float = 2e-4
    momentum: float = 0.9
    weight_decay: float = 1e-5
    schedule: Literal['constant', '1cycle', 'exponential', 'cosine', 'step', 'reduceonplateau'] = 'constant'
    completed_epochs: int = 0
    augment: bool = False
    step_size: int = 10
    gamma: float = 0.1
    rop_factor: float = 0.1
    rop_patience: int = 5
    cos_t_max: int = 50
    cos_min_lr: float = 2e-5
    warmup: int = 0


class Trainer():
    """
    Class for training or finetuning a Kraken segmentation model.
    """
    
    def __init__(
        self, 
        data_config: DataConfig,
        trainer_config: TrainerConfig,
        device: str = 'auto',
        precision: Literal['16', '16-mixed', '32', '32-true', '64', '64-true', 'bf16', 'bf16-mixed'] = '32-true',
        threads: int = 1,
        workers: int = 1,
        seed: int | None = None,
        **kwargs
    ) -> None:
        self.data_config = data_config
        self.trainer_config = trainer_config
        self.device = device
        self.precision = precision
        self.threads = threads
        
        logging.captureWarnings(True)
        logging.getLogger('lightning.fabric.utilities.seed').setLevel(logging.ERROR)
        
        if seed is not None:
            logger.info(f'Seed set to {seed}')
            seed_everything(seed, workers=True)
        elif data_config.deterministic:
            logger.info('Seed set to 42')
            seed_everything(42, workers=True)
        
        checkpoints = Path(data_config.output) / 'checkpoints'
        checkpoints.mkdir(exist_ok=True, parents=True)
        
        logger.info('Update hyperparameters')
        if trainer_config.resize != 'fail' and not data_config.model:
            raise ValueError('Resize != \'fail\' requires loading an existing model')
        if not (0 <= trainer_config.frequency <= 1) and trainer_config.frequency % 1.0 != 0:
            raise ValueError('Frequency needs to be either in the interval [0.0, 1.0] or a positive integer')
        
        hyper_params = SEGMENTATION_HYPER_PARAMS.copy()
        hyper_params.update({
            'line_width': trainer_config.line_width,
            'padding': trainer_config.padding,
            'freq': trainer_config.frequency,
            'quit': trainer_config.quit,
            'epochs': trainer_config.epochs,
            'min_epochs': trainer_config.min_epochs,
            'lag': trainer_config.lag,
            'optimizer': trainer_config.optimizer,
            'lrate': trainer_config.lrate,
            'momentum': trainer_config.momentum,
            'weight_decay': trainer_config.weight_decay,
            'schedule': trainer_config.schedule,
            'completed_epochs': trainer_config.completed_epochs,
            'augment': trainer_config.augment,
            'step_size': trainer_config.step_size,
            'gamma': trainer_config.gamma,
            'rop_factor': trainer_config.rop_factor,
            'rop_patience': trainer_config.rop_patience,
            'cos_t_max': trainer_config.cos_t_max,
            'cos_min_lr': trainer_config.cos_min_lr,
            'warmup': trainer_config.warmup,
        })
        if hyper_params['freq'] > 1:  # ty:ignore[unsupported-operator]
            self.trainer_params = {
                'check_val_every_n_epoch': int(hyper_params['freq'])  # ty:ignore[invalid-argument-type]
            }
        else:
            self.trainer_params = {
                'val_check_interval': float(hyper_params['freq'])  # ty:ignore[invalid-argument-type]
            }
        
        custom_attributes = {}
        try:
            from octopy.plugins import OctopyTrainer
            OctopyTrainer.register()
            custom_attributes['image_extension'] = data_config.image_extension
        except ImportError as exc:
            logger.warning(f'Could not install custom SegmentationModel: {str(exc)}')
            
        
        self.segmentation_model = SegmentationModel(
            hyper_params=hyper_params,
            load_hyper_parameters=data_config.model is not None,
            output=checkpoints.joinpath(data_config.model_name).as_posix(),
            spec=trainer_config.vgsl,
            model=data_config.model,
            training_data=data_config.train_data,
            evaluation_data=data_config.eval_data,
            partition=1 if data_config.eval_data else data_config.partition,
            num_workers=workers,
            format_type='page',
            suppress_regions=self.data_config.suppress_regions,
            suppress_baselines=self.data_config.suppress_baselines,
            valid_regions=self.data_config.valid_regions or None,
            valid_baselines=self.data_config.valid_regions or None,
            merge_regions=None if self.data_config.merge_regions is None 
                          else self._build_merge_dict(self.data_config.merge_regions),
            merge_baselines=None if self.data_config.merge_baselines is None 
                            else self._build_merge_dict(self.data_config.merge_baselines),
            resize=trainer_config.resize,
            topline={'baseline': False, 'topline': True}.get(data_config.position, None),
            **custom_attributes  # ty:ignore[invalid-argument-type]
        )
        
        table = Table()#title='File Summary')
        table.add_column('Partition')
        table.add_column('Count', justify='right')
        table.add_row('Training', str(len(self.segmentation_model.train_set)))
        table.add_row('Evaluation', str(len(self.segmentation_model.val_set)))
        if (spinner := kwargs.get('console', None)) is not None:
            spinner.console.print(table)
        else:
            Console().print(table)
        
        table = Table()#title='Class Summary')
        table.add_column('Category')
        table.add_column('Class')
        table.add_column('ID', justify='right')
        table.add_column('Merged With')
        table.add_column('Count', justify='right')
        for section in ('baselines', 'regions'):
            dataset = self.segmentation_model.train_set.dataset
            for cls, idx in dataset.class_mapping[section].items():  # ty:ignore[unresolved-attribute]
                if section == 'baselines' and self.data_config.merge_baselines is not None:
                    merged = self.data_config.merge_baselines.get(cls, [])
                elif section == 'regions' and self.data_config.merge_regions is not None:
                    merged = self.data_config.merge_regions.get(cls, [])
                else:
                    merged = []
                count = dataset.class_stats[section][cls]  # ty:ignore[unresolved-attribute]
                table.add_row(section, cls, str(idx), ', '.join(merged), str(count))
        if (spinner := kwargs.get('console', None)) is not None:
            spinner.console.print(table, end='\n\n')
        else:
            Console().print(table, end='\n\n')
            
    def fit(self) -> None:
        logger.info('Build lightning trainer') 
        accelerator, devices = self._parse_device(self.device)       
        trainer = KrakenTrainer(
            accelerator=accelerator,
            devices=devices,
            precision=self.precision,
            max_epochs=self.trainer_config.epochs if self.trainer_config.quit == 'fixed' else -1,
            min_epochs=self.trainer_config.min_epochs,
            enable_progress_bar=True,
            deterministic=self.data_config.deterministic,
            **self.trainer_params  # ty:ignore[invalid-argument-type]
        )
        
        with threadpool_limits(limits=self.threads):
            trainer.fit(self.segmentation_model)
            
        logger.info('Evaluate results')
        if self.segmentation_model.best_epoch == -1:
            print('WARNING: Model did not improve during training')
            return
        print(f'Best model found at epoch {self.segmentation_model.best_epoch} '
              f'with metric {self.segmentation_model.best_metric}')
        best_model_path = self.segmentation_model.best_model
        if best_model_path is None:
            raise RuntimeError('No best model found')
        outfile = Path(self.data_config.output) / f'{self.data_config.model_name}_best.mlmodel'
        copy(Path(best_model_path), outfile)
        print(f'Saved to {outfile.as_posix()}')
    
    @staticmethod
    def _parse_device(device: str) -> tuple[str, str | list[int]]:
        """
        Parses the input device string to a pytorch accelerator and device string.
        Args:
            device: Encoded device string (see PyTorch documentation).
        Returns:
            Tuple containing accelerator string and device integer/string.
        """
        auto_devices = ['auto', 'cpu', 'mps']
        acc_devices = ['cuda', 'tpu', 'hpu', 'ipu']
        if device in auto_devices:
            return device, 'auto'
        elif any([device.startswith(x) for x in acc_devices]):
            dv, i = device.split(':')
            if dv == 'cuda':
                dv = 'gpu'
            return dv, [int(i)]
        else:
            raise ValueError(f'Invalid device string: {device}')
        
    @staticmethod
    def _build_merge_dict(merge: dict[str, list[str]]) -> dict[str, str]:
        rules: dict[str, str] = {}
        for key, value in merge.items():
            for v in value:
                if v in rules:
                    raise ValueError(f'Source class cannot be merged into multiple target classes: {v}')
                if v in merge.keys():
                    raise ValueError(f'Nested merges are not allowed: {v}')
                rules[v] = key
        return rules
