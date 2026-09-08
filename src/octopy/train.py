# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Literal

from kraken.configs import (BLLASegmentationTrainingConfig,
                            BLLASegmentationTrainingDataConfig)
from kraken.models import convert_models
from kraken.train import (BLLASegmentationDataModule,
                          BLLASegmentationModel,
                          KrakenTrainer)
from kraken.train.utils import KrakenOnExceptionCheckpoint
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
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
    position: Literal['baseline', 'topline', 'centerline'] = 'baseline'
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
    optimizer: Literal['Adam', 'AdamW', 'AdamW+Muon', 'SGD', 'RMSprop'] = 'AdamW'
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
    weights_format: Literal['safetensors', 'coreml'] = 'safetensors'


class _Counter:
    """Auto-incrementing label counter starting at 2 (0/1 are reserved for aux classes)."""
    def __init__(self, start: int = 2) -> None:
        self.n = start

    def __call__(self) -> int:
        val = self.n
        self.n += 1
        return val


class _ClassMapping(defaultdict):
    """
    A class mapping that filters classes against `valid` and remaps merged
    sources onto their target's label index on first access.
    """
    def __init__(self, counter: _Counter, valid: list[str] | None = None, merge: dict[str, str] | None = None) -> None:
        super().__init__(counter)
        self._valid: set[str] | None = set(valid) if valid else None
        self._merge: dict[str, str] = merge or {}

    def __missing__(self, key: str) -> int:
        target = self._merge.get(key, key)
        if self._valid is not None and target not in self._valid:
            raise KeyError(key)
        if target not in self:
            self[target] = self.default_factory()
        return self[target]


class Trainer():
    """
    Class for training or finetuning a Kraken segmentation model.
    """
    
    def __init__(
        self, 
        data_config: DataConfig,
        trainer_config: TrainerConfig,
        device: str = 'auto',
        precision: str = '32-true',
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
        
        logger.info('Update hyperparameters')
        if trainer_config.resize != 'fail' and not data_config.model:
            raise ValueError('Resize != \'fail\' requires loading an existing model')
        if not (0 <= trainer_config.frequency <= 1) and not trainer_config.frequency.is_integer():
            raise ValueError('Frequency needs to be either in the interval [0.0, 1.0] or a positive integer')
        
        checkpoints = Path(data_config.output) / 'checkpoints' / data_config.model_name
        checkpoints.mkdir(exist_ok=True, parents=True)
        
        try:
            from octopy.plugins import OctopyTrainer
            OctopyTrainer.register()
        except ImportError as exc:
            logger.warning(f'Could not install custom DataModule: {str(exc)}')
        
        counter = _Counter(2)
        data_module_config = BLLASegmentationTrainingDataConfig(
            training_data=[str(x) for x in data_config.train_data],
            evaluation_data=[str(x) for x in data_config.eval_data] if data_config.eval_data else None,
            partition=1 if data_config.eval_data else data_config.partition,
            num_workers=workers,
            augment=trainer_config.augment,
            format_type='page',
            line_width=trainer_config.line_width,
            topline={'baseline': False, 'topline': True, 'centerline': None}.get(data_config.position, None),
            line_class_mapping=self._build_class_mapping(data_config.suppress_baselines,
                                                         data_config.valid_baselines,
                                                         data_config.merge_baselines,
                                                         counter),
            region_class_mapping=self._build_class_mapping(data_config.suppress_regions,
                                                           data_config.valid_regions,
                                                           data_config.merge_regions,
                                                           counter),
        )
        data_module_config.image_extension = data_config.image_extension
        self.data_module = BLLASegmentationDataModule(data_module_config)

        self.model_config = BLLASegmentationTrainingConfig(
            spec=trainer_config.vgsl,
            padding=trainer_config.padding,
            resize=trainer_config.resize,
            freq=trainer_config.frequency,
            checkpoint_path=checkpoints.as_posix(),
            weights_format=trainer_config.weights_format,
            quit=trainer_config.quit,
            epochs=trainer_config.epochs,
            min_epochs=trainer_config.min_epochs,
            lag=trainer_config.lag,
            optimizer=trainer_config.optimizer,
            lrate=trainer_config.lrate,
            momentum=trainer_config.momentum,
            weight_decay=trainer_config.weight_decay,
            schedule=trainer_config.schedule,
            completed_epochs=trainer_config.completed_epochs,
            step_size=trainer_config.step_size,
            gamma=trainer_config.gamma,
            rop_factor=trainer_config.rop_factor,
            rop_patience=trainer_config.rop_patience,
            cos_t_max=trainer_config.cos_t_max,
            cos_min_lr=trainer_config.cos_min_lr,
            warmup=trainer_config.warmup,
        )
        
        table = Table()#title='File Summary')
        table.add_column('Partition')
        table.add_column('Count', justify='right')
        table.add_row('Training', str(len(self.data_module.train_set)))
        table.add_row('Evaluation', str(len(self.data_module.val_set)))
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
        dataset = self.data_module.train_set.dataset
        canonical = dataset.canonical_class_mapping
        merged = dataset.merged_classes
        for section in ('baselines', 'regions'):
            for cls, idx in canonical[section].items():
                aliases = merged[section].get(cls, [])
                count = dataset.class_stats[section].get(cls, 0)
                for alias in aliases:
                    count += dataset.class_stats[section].get(alias, 0)
                table.add_row(section, cls, str(idx), ', '.join(aliases), str(count))
        if (spinner := kwargs.get('console', None)) is not None:
            spinner.console.print(table, end='\n\n')
        else:
            Console().print(table, end='\n\n')
            
    def fit(self) -> None:
        logger.info('Build lightning trainer')
        accelerator, devices = self._parse_device(self.device)

        if self.model_config.freq > 1:
            val_check_interval = {'check_val_every_n_epoch': int(self.model_config.freq)}
        else:
            val_check_interval = {'val_check_interval': self.model_config.freq}

        checkpoint_callback = ModelCheckpoint(
            dirpath=self.model_config.checkpoint_path,
            save_top_k=10,
            monitor='val_metric',
            mode='max',
            auto_insert_metric_name=False,
            filename='checkpoint_{epoch:02d}-{val_metric:.4f}'
        )
        callbacks = [KrakenOnExceptionCheckpoint(dirpath=self.model_config.checkpoint_path,
                                                 filename='checkpoint_abort'),
                     checkpoint_callback]

        trainer = KrakenTrainer(
            accelerator=accelerator,
            devices=devices,
            precision=self.precision,
            max_epochs=self.trainer_config.epochs if self.trainer_config.quit == 'fixed' else -1,
            min_epochs=self.trainer_config.min_epochs,
            enable_progress_bar=True,
            deterministic=self.data_config.deterministic,
            callbacks=callbacks,
            num_sanity_val_steps=0,
            use_distributed_sampler=False,
            **val_check_interval
        )
        
        with trainer.init_module(empty_init=False if self.data_config.model else True):
            if self.data_config.model:
                logger.info(f'Loading model from {self.data_config.model}')
                model = BLLASegmentationModel.load_from_weights(self.data_config.model, config=self.model_config)
            else:
                model = BLLASegmentationModel(self.model_config)

        with threadpool_limits(limits=self.threads):
            trainer.fit(model, self.data_module)
            
        logger.info('Evaluate results')
        if checkpoint_callback.best_model_path is None:
            print('WARNING: Model did not improve during training')
            return
        score = checkpoint_callback.best_model_score.item()
        print(f'Best model found with metric {score:.4f}')
        outfile = Path(self.data_config.output) / f'{self.data_config.model_name}_best.{self.trainer_config.weights_format}'
        opath = convert_models([checkpoint_callback.best_model_path], outfile, weights_format=self.trainer_config.weights_format)
        print(f'Saved to {Path(opath).as_posix()}')

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

    @classmethod
    def _build_class_mapping(cls,
                             suppress: bool,
                             valid: list[str] | None,
                             merge: dict[str, list[str]] | None,
                             counter: _Counter) -> dict:
        if suppress:
            return {}
        if valid is None and not merge:
            return defaultdict(counter)
        return _ClassMapping(counter, valid=valid, merge=cls._build_merge_dict(merge or {}))
