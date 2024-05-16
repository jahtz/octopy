from pathlib import Path
import shutil
import logging

import click
from kraken.lib.train import SegmentationModel, KrakenTrainer
from kraken.lib.default_specs import SEGMENTATION_HYPER_PARAMS, SEGMENTATION_SPEC
from kraken.lib import log

from modules.helper import path_parser


__all__ = ['blstrain_workflow']

AUTO_DEVICES = ['cpu', 'mps']
ACC_DEVICES = ['cuda', 'tpu', 'hpu', 'ipu']
VERBOSE = 0

logging.captureWarnings(True)
logger = logging.getLogger('kraken')
log.set_logger(logger, level=30 - min(10 * VERBOSE, 20))


def _device_parser(device: str) -> tuple:
    if device in AUTO_DEVICES:
        return device, 'auto'
    elif any([device.startswith(x) for x in ACC_DEVICES]):
        d, i = device.split(':')
        if d == 'cuda':
            d = 'gpu'
        return d, [int(i)]
    else:
        raise Exception(f'Unknown device: {device}')


def blstrain_workflow(
        _input: tuple,
        output_path: Path,
        _train: tuple | None = None,
        _eval: tuple | None = None,
        output_name: str = 'foo',
        eval_percentage: int = 10,
        device: str = 'cpu',
        threads: int = 1,
        base_model: Path | None = None,
        train_regions: bool = True,
        train_lines: bool = True,
        max_epochs: int = 50,
        min_epochs: int = 0,
):
    """
    Train baseline segmentation model.

    :param _input: paths to ground truth files
    :param _train: paths to additional training files
    :param _eval: paths to evaluation files
    :param eval_percentage: percentage of ground truth data used for evaluation
    :param device: computation device
    :param output_path: path to output directory
    :param output_name: name of output model
    :param threads: number of allocated threads
    :param base_model: model to start training from
    :param train_regions: enable region training
    :param train_lines: enable baseline training
    :param max_epochs: max epochs
    :param min_epochs: min epochs
    :return: nothing
    """
    # load ground truth files
    ground_truth = path_parser(_input)

    if _train and (tf := path_parser(_train)):
        ground_truth.extend(tf)

    if not ground_truth:
        click.echo('No ground truth data found. Exiting...', err=True)
        return

    # load evaluation files
    if _eval:
        evaluation = path_parser(_eval)
    else:
        evaluation = None

    # calculate partition percentage
    partition = 1.0 if _eval else (1.0 - (eval_percentage / 100.0))

    # device selection
    if device in AUTO_DEVICES:
        accelerator = device
        devices = 'auto'
    elif any([device.startswith(x) for x in ACC_DEVICES]):
        accelerator, i = device.split(':')
        if accelerator == 'cuda':
            accelerator = 'gpu'
        devices = [int(i)]
    else:
        click.echo(f'Unknown device: {device}', err=True)
        return

    # create output directory
    cp_path = output_path.joinpath('checkpoints')
    cp_path.mkdir(parents=True, exist_ok=True)

    # load hyperparameters
    hyper_params = SEGMENTATION_HYPER_PARAMS.copy()

    # create training model
    model = SegmentationModel(
        hyper_params,
        spec=SEGMENTATION_SPEC,
        output=cp_path.joinpath(output_name).as_posix(),
        model=base_model,
        training_data=ground_truth,
        evaluation_data=evaluation,
        partition=partition,
        num_workers=threads,
        load_hyper_parameters=True,
        format_type='page',
        suppress_regions=not train_regions,
        suppress_baselines=not train_lines,
        resize='both',
    )

    # check if training data is available
    if len(model.train_set) == 0:
        click.echo('No training data found. Exiting...', err=True)
        return

    # create trainer
    trainer = KrakenTrainer(
        accelerator=accelerator,
        devices=devices,
        precision='32',
        max_epochs=max_epochs,
        min_epochs=min_epochs,
        enable_progress_bar=True,
        pl_logger=None,
        val_check_interval=1.0,
        deterministic=False,
    )

    # start training
    trainer.fit(model)

    # check if model improved and save best model
    if model.best_epoch == -1:
        click.echo('Model did not improve during training. Exiting...', err=True)
        return
    click.echo(f'Best model found at epoch {model.best_epoch}')
    best_model_path = model.best_model
    shutil.copy(best_model_path, output_path.joinpath(f'{output_name}_best.mlmodel'))
    click.echo(f'Saved to: {output_path.joinpath(f"{output_name}_best.mlmodel")}')
