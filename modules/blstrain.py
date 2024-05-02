from pathlib import Path
import shutil

import click
from kraken.lib.train import SegmentationModel, KrakenTrainer
from kraken.lib.default_specs import SEGMENTATION_HYPER_PARAMS

AUTO_DEVICES = ['cpu', 'mps']
ACC_DEVICES = ['cuda', 'tpu', 'hpu', 'ipu']


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


def _page_to_training_data(page: Path) -> dict:
    """
    Parsing PageXML file to training data.

    :param page: path to PageXML
    :return: training data dictionary
    """
    pass  # TODO: implement for custom image directory


def blstrain_workflow(
        gt_files: Path,
        gt_regex: str = '*.xml',
        training_files: Path | None = None,
        training_regex: str = '*.xml',
        eval_files: Path | None = None,
        eval_regex: str = '*.xml',
        eval_percentage: int = 10,
        device: str = 'cpu',
        output_path: Path | None = None,
        output_name: str = 'foo',
        threads: int = 1,
        base_model: Path | None = None,
        train_regions: bool = True,
        train_lines: bool = True,
        max_epochs: int = 50,
        min_epochs: int = 0,
):
    """
    Train baseline segmentation model.

    :param gt_files: path to ground truth files
    :param gt_regex: regex for ground truth files
    :param training_files: path to additional training files
    :param training_regex: regex for additional training files
    :param eval_files: path to evaluation files
    :param eval_regex: regex for evaluation files
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
    ground_truth = list([fp.as_posix() for fp in gt_files.glob(gt_regex)])
    if training_files is not None and (tf := list([fp.as_posix() for fp in training_files.glob(training_regex)])):
        ground_truth.extend(tf)
    if not ground_truth:
        click.echo('No ground truth data found. Exiting...', err=True)
        return

    # load evaluation files
    if eval_files is not None:
        evaluation = list([fp.as_posix() for fp in eval_files.glob(eval_regex)])
    else:
        evaluation = None

    # calculate partition percentage
    partition = (1.0 - (eval_percentage / 100.0)) if eval_files is None else 1.0

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

    if len(model.train_set) == 0:
        click.echo('No training data found. Exiting...', err=True)
        return

    trainer = KrakenTrainer(
        accelerator=accelerator,
        devices=devices,
        precision='32',
        max_epochs=max_epochs,
        min_epochs=min_epochs,
        enable_progress_bar=True,
        pl_logger=None,
        val_check_interval=1.0,
    )

    trainer.fit(model)

    if model.best_epoch == -1:
        click.echo('Model did not improve during training. Exiting...', err=True)
        return
    else:
        click.echo(f'Best model found at epoch {model.best_epoch}')

    best_model_path = model.best_model
    shutil.copy(best_model_path, output_path.joinpath(f'{output_name}_best.mlmodel'))
    click.echo(f'Best model saved to: {output_path.joinpath(f"{output_name}_best.mlmodel")}')


if __name__ == '__main__':
    blstrain_workflow(

    )

