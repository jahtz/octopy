# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations
from ty_extensions import Unknown

import logging
from pathlib import Path
from typing import Literal, Any

import click
from threadpoolctl import threadpool_limits
from lightning.pytorch.callbacks import ModelCheckpoint
from rich.console import Console
from rich.table import Table
from kraken.lib import vgsl  # NOQA
from kraken.train import KrakenTrainer, BLLASegmentationDataModule, BLLASegmentationModel
from kraken.train.utils import KrakenOnExceptionCheckpoint
from kraken.configs import BLLASegmentationTrainingConfig, BLLASegmentationTrainingDataConfig
from kraken.models.convert import convert_models


logger: logging.Logger = logging.getLogger(__name__)
#logging.captureWarnings(True)
#logging.getLogger("lightning.fabric.utilities.seed").setLevel(logging.ERROR)
#install(suppress=[click])


def generate_data_config(
    training_data: list[Path],
    evaluation_data: list[Path] | None = None,
    test_data: list[Path] | None = None,
    partition: float = 0.9,
    num_workers: int = 1,
    augment: bool = False,
    batch_size: int = 1,
    line_width: int = 8,
    topline: bool | None = False,
) -> BLLASegmentationTrainingDataConfig:
    """
    Set training data configuration.
    Args:
        training_data: A list of training PAGE-XML files.
        evaluation_data: An optional list of evaluation PAGE-XML files. Defaults to None.
        test_data: An optional list of test PAGE-XML files. Defaults to None.
        partition: Automatic partition of training data files if no evaluation data is defined. Defaults to 0.9.
        num_workers: Number of dataloader workers. Defaults to 1.
        augment: Switch to enable augmentation. Defaults to False.
        batch_size: Number of items to pack into a single sample. Defaults to 1.
        line_width: ine width in the target segmentation map. Defaults to 8.
        topline: Indicator of baseline position in dataset. False = baseline, True = topline, None = centerline. 
            Defaults to False.
    """
    return BLLASegmentationTrainingDataConfig(
        training_data=training_data,
        evaluation_data=evaluation_data,
        test_data=test_data,
        partition=partition,
        num_workers=num_workers,
        augment=augment,
        batch_size=batch_size,
        line_width=line_width,
        topline=topline,
        format_type='page'
    )


def generate_model_config(
    spec: str = '[1,1800,0,3 Cr7,7,64,2,2 Gn32 Cr3,3,128,2,2 Gn32 Cr3,3,128 Gn32 Cr3,3,256 Gn32 Cr3,3,256 Gn32 Lbx32 Lby32 Cr1,1,32 Gn32 Lby32 Lbx32]',
    padding: tuple[int, int] = (0, 0),
    resize: Literal['union', 'new', 'fail'] = 'new',
    bl_tol: float = 10.0,
    dice_weight: float = 0.5,
    epochs: int = -1,
    completed_epochs: int = 0,
    freq: float = 1.0,
    checkpoint_path: str = 'model',
    weights_format: Literal['safetensors', 'coreml'] = 'safetensors',
    optimizer: Literal['Adam', 'AdamW', 'SGD', 'RMSprop'] = 'AdamW',
    lrate: float = 1e-5,
    momentum: float = 0.9,
    weight_decay: float = 0.0,
    gradient_clip_val: float = 1.0,
    accumulate_grad_batches: int = 1,
    schedule: Literal['cosine', 'constant', 'exponential', 'step', '1cycle', 'reduceonplateau'] = 'constant',
    warmup: int = 0,
    step_size: int = 10,
    gamma: float = 0.1,
    rop_factor: float = 0.1,
    rop_patience: int = 5,
    cos_t_max: int = 10,
    cos_min_lr: float = 1e-6,
    quit: Literal['early', 'fixed'] = 'early',
    min_epochs: int = 0,
    lag: int = 10,
    min_delta: float = 0.0,
    precision: Literal['transformer-engine', 'transformer-engine-float16', '16-true', '16-mixed', 'bf16-true', 'bf16-mixed', '32-true', '64-true'] = '32-true',
    accelerator: str = 'auto',
    device: str = 'auto',
    batch_size: int = 1,
    compile_config: dict[str, Any] | None = None,
    raise_on_error: bool = False,
    num_threads: int = 1
) -> BLLASegmentationTrainingConfig:
    """
    Set training model configuration.
    Args:
        spec: VGSL model description. Defaults to 
            '[1,1800,0,3 Cr7,7,64,2,2 Gn32 Cr3,3,128,2,2 Gn32 Cr3,3,128 Gn32 Cr3,3,256 Gn32 Cr3,3,256 Gn32 Lbx32 Lby32 Cr1,1,32 Gn32 Lby32 Lbx32]'.
        padding: Padding (left/right, top/bottom) around the page image. Defaults to (0, 0).
        resize: Controls how the model's output layer is resized if the training data contains different classes.
            union` adds new classes (former `add`), `new` resizes to match the training data (former `both`), 
            and `fail` aborts training if there is a mismatch. Defaults to 'new'.
        bl_tol: Tolerance in pixels for baseline detection metrics. Defaults to 10.0.
        dice_weight: No documentation. Defaults to 0.5.
        epochs: Number of epochs to train for when using fixed stopping. Defaults to -1.
        completed_epochs: How many epochs of the schedule have already been completed. Defaults to 0.
        freq: Model saving and report generation frequency in epochs during training. If frequency is >1 it must be 
            an integer, i.e. running validation every n-th epoch. Defaults to 1.0.
        checkpoint_path: # Path prefix to save checkpoints during training. Defaults to 'model'.
        weights_format: Weight format to convert checkpoint at end of training to. Defaults to 'safetensors'.
        optimizer: Optimizer to use. Defaults to 'AdamW'.
        lrate: Learning rate. Defaults to 1e-5.
        momentum: Momentum parameter. Ignored if optimizer doesn't use it. Defaults to 0.9.
        weight_decay: Weight decay. Ignored if optimizer doesn't support it. Defaults to 0.0.
        gradient_clip_val: Threshold for gradient clipping. Defaults to 1.0.
        accumulate_grad_batches: Number of batches to aggregate before backpropagation. Defaults to 1.
        schedule: Type of learning rate schedule. Defaults to 'constant'.
        warmup: Number of iterations to warmup learning rate. Defaults to 0.
        step_size: Learning rate decay in stepped schedule. Defaults to 10.
        gamma: Learning rate decay in exponential schedule. Defaults to 0.1.
        rop_factor: Learning rate decay in reduce on plateau schedule. Defaults to 0.1.
        rop_patience: Number of epochs to wait before reducing learning rate. Defaults to 5.
        cos_t_max: Epoch at which cosine schedule reaches final learning rate. Defaults to 10.
        cos_min_lr: Final learning rate with cosine schedule. Defaults to 1e-6.
        quit: Stop condition for training. Choose `early` for early stopping or `fixed` for a fixed number of 
            epochs. Defaults to 'early'.
        min_epochs: Minimum number of epochs to train without considering validation scores. Defaults to 0.
        lag: Number of epochs to wait for improvement in validation scores before aborting. Defaults to 10.
        min_delta: Minimum delta of validation scores. Defaults to 0.0.
        precision: Sets the precision to run the model in. Defaults to '32-true'.
        accelerator: No documentation. Defaults to 'auto'.
        device: No documentation. Defaults to 'auto'.
        batch_size: Sets the batch size for inference. Defaults to 1.
        compile_config: Decides how kraken will compile the forward pass of the model. If not given compilation will 
            be disabled. To enable with default parameters set an empty dictionary. Defaults to None.
        raise_on_error: Causes an exception to be raised instead of internal handling when functional blocks that 
            can fail for misshapen input crash. Defaults to False.
        num_threads: Number of threads to use for intra-op parallelisation. Defaults to 1.
    """
    return BLLASegmentationTrainingConfig(
        spec=spec,
        padding=padding,
        resize=resize,
        bl_tol=bl_tol,
        dice_weight=dice_weight,
        epochs=epochs,
        completed_epochs=completed_epochs,
        freq=freq,
        checkpoint_path=checkpoint_path,
        weights_format=weights_format,
        optimizer=optimizer,
        lrate=lrate,
        momentum=momentum,
        weight_decay=weight_decay,
        gradient_clip_val=gradient_clip_val,
        accumulate_grad_batches=accumulate_grad_batches,
        schedule=schedule,
        warmup=warmup,
        step_size=step_size,
        gamma=gamma,
        rop_factor=rop_factor,
        rop_patience=rop_patience,
        cos_t_max=cos_t_max,
        cos_min_lr=cos_min_lr,
        quit=quit,
        min_epochs=min_epochs,
        lag=lag,
        min_delta=min_delta,
        precision=precision,
        accelerator=accelerator,
        device=device,
        batch_size=batch_size,
        compile_config=compile_config,
        raise_on_error=raise_on_error,
        num_threads=num_threads
    )


class Trainer:
    def __init__(
        self, 
        deterministic: bool = True,
        seed: int | None = None,
    ) -> None:
        """
        Initialize segmentation trainer.
        Args:
            resume: Load a checkpoint to continue training. Defaults to None.
            deterministic: Enables deterministic training. If no seed is given and enabled the seed will be set to 42.
            seed: Seed for numpy\'s and torch\'s RNG. Set to a fixed value to ensure reproducible random splits of data.
        """
        self.deterministic: bool | str = False if not deterministic else 'warn'
        if seed is not None:
            from lightning.pytorch import seed_everything
            seed_everything(seed, workers=True)
        elif deterministic:
            from lightning.pytorch import seed_everything
            seed_everything(42, workers=True)
        
        self.model_config: BLLASegmentationTrainingConfig | None = None
        self.data_config: BLLASegmentationTrainingDataConfig | None = None
        self.data_module: BLLASegmentationDataModule | None = None
        self.trainer: KrakenTrainer | None = None
        self.model: BLLASegmentationModel | None = None

    def initialize(
        self, 
        model_config: BLLASegmentationTrainingConfig,
        data_config: BLLASegmentationTrainingDataConfig | None = None,
        load: Path | None = None,
        resume: Path | None = None,
    ) -> None:
        """
        Initialize Kraken segmentation training.
        Args:
            load: Load existing file to continue training. Defaults to None.
            resume: Load a checkpoint to continue training. Defaults to None.
            data_config: _description_. Defaults to None.
            model_config: _description_. Defaults to None.

        Raises:
            ValueError: _description_
            click.UsageError: _description_
        """
        if sum(map(bool, [resume, load])) > 1:
            raise click.UsageError('load/resume options are mutually exclusive.')
        
        if model_config.freq > 1:
            val_check_interval: dict[str, int] = {'check_val_every_n_epoch': int(model_config.freq)}
        else:
            val_check_interval: dict[str, float] = {'val_check_interval': model_config.freq}
        
        cbs: list[Any] = [
            KrakenOnExceptionCheckpoint(
                dirpath=model_config.checkpoint_path,
                filename='checkpoint_abort'
            ),
            ModelCheckpoint(
                dirpath=model_config.checkpoint_path,
                save_top_k=10,
                monitor='val_metric',
                mode='max',
                auto_insert_metric_name=False,
                filename='checkpoint_{epoch:02d}-{val_metric:.4f}'
            )
        ]
        
        if resume:
            self.data_module: BLLASegmentationDataModule = BLLASegmentationDataModule.load_from_checkpoint(
                resume, 
                weights_only=False
            )
        else:
            if data_config is None:
                raise click.UsageError('data_config required for ')
            
            if data_config.evaluation_data:
                data_config.partition = 1
            
            if len(data_config.training_data) == 0:
                raise click.UsageError('No training data was provided. Use \'-g\' to add PAGE-XML files.')
            self.data_module: BLLASegmentationDataModule = BLLASegmentationDataModule(data_config)

        # load classes
        ds = self.data_module.train_set.dataset
        canonical = ds.canonical_class_mapping  # ty:ignore[unresolved-attribute]
        merged = ds.merged_classes  # ty:ignore[unresolved-attribute]

        table = Table(title='Training Class Summary')
        table.add_column('Category')
        table.add_column('Class')
        table.add_column('Label Index', justify='right')
        table.add_column('Merged With')
        table.add_column('Count', justify='right')

        for section in ('baselines', 'regions'):
            for cls_name, idx in canonical[section].items():
                aliases = merged[section].get(cls_name, [])
                merged_str = ', '.join(aliases) if aliases else ''
                count = ds.class_stats[section].get(cls_name, 0)  # ty:ignore[unresolved-attribute]
                for alias in aliases:
                    count += ds.class_stats[section].get(alias, 0)  # ty:ignore[unresolved-attribute]
                table.add_row(section, cls_name, str(idx), merged_str, str(count))

        Console(stderr=True).print(table)

        self.trainer: KrakenTrainer = KrakenTrainer(
            accelerator=model_config.accelerator,
            devices=model_config.device,
            precision=model_config.precision,
            max_epochs=model_config.epochs if model_config.quit == 'fixed' else -1,
            min_epochs=model_config.min_epochs,
            enable_progress_bar=True,
            deterministic=self.deterministic,
            enable_model_summary=False,
            accumulate_grad_batches=model_config.accumulate_grad_batches,
            callbacks=cbs,
            gradient_clip_val=model_config.gradient_clip_val,
            num_sanity_val_steps=0,
            use_distributed_sampler=False,
            **val_check_interval  # ty:ignore[invalid-argument-type]
        )
        
        with self.trainer.init_module(empty_init=False if (load or resume) else True):
            if load:
                logger.info(f'Loading from checkpoint {load}.')
                if load.name.endswith('ckpt'):
                    self.model: BLLASegmentationModel = BLLASegmentationModel.load_from_checkpoint(load, config=m_config, weights_only=False)
                else:
                    self.model: BLLASegmentationModel = BLLASegmentationModel.load_from_weights(load, config=model_config)
            elif resume:
                logger.info(f'Resuming from checkpoint {resume}.')
                self.model: BLLASegmentationModel = BLLASegmentationModel.load_from_checkpoint(resume, weights_only=False)
            else:
                logger.info('Initializing new model.')
                self.model: BLLASegmentationModel = BLLASegmentationModel(model_config)
                

    def run(self) -> None:
        if self.model is None or self.trainer is None or self.model_config is None:
            raise click.UsageError('The trainer has to be initialized first')
        
        with threadpool_limits(limits=self.model_config.num_threads):
            if resume:
                self.trainer.fit(self.model, self.data_module, ckpt_path=resume)
            else:
                self.trainer.fit(self.model, self.data_module)

        score = checkpoint_callback.best_model_score.item()
        weight_path = Path(checkpoint_callback.best_model_path).with_name(f'best_{score:.4f}.{params.get("weights_format")}')
        opath = convert_models([checkpoint_callback.best_model_path], weight_path, weights_format=params['weights_format'])
        message(f'Converting best model {checkpoint_callback.best_model_path} (score: {score:.4f}) to weights {opath}')
    