from pathlib import Path
import shutil

import click
import numpy as np
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
        xmls: Path,
        output_path: Path,
        output_name: str = 'foo',
        regex: str = '*.xml',
        base_model: Path | None = None,
        eval_percentage: int = 20,
        threads: int = 1,
        device: str = 'cpu',
        max_epochs: int = 300,
        min_epochs: int = 5,
):
    """
    Train Kraken segmentation model.

    :param xmls: ground truth PageXML files
    :param output_path: path for models and checkpoints.
    :param output_name: name for best model after training.
    :param regex: regex for ground truth data selection.
    :param base_model: model to train from.
    :param eval_percentage: percentage of ground truth used for evaluation.
    :param threads: number of worker threads.
    :param device: device for computation. CUDA recommended: 'cuda:0'.
    :param max_epochs: maximal number of epochs.
    :param min_epochs: minimal number of epochs.
    :return: nothing.
    """
    # prepare training and evaluation data
    files = list(x.as_posix() for x in xmls.glob(regex))
    np.random.default_rng(6546513218165156132186165165).shuffle(files)
    eval_index = max(1, int(len(files) * eval_percentage / 100))
    training_data = files[eval_index:]
    evaluation_data = files[:eval_index]

    # create output directory
    checkpoint_path = output_path.joinpath('checkpoints')
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    # create training model
    training_model = SegmentationModel(
        SEGMENTATION_HYPER_PARAMS,
        load_hyper_parameters=True,
        output=checkpoint_path.joinpath(output_name).as_posix(),
        model=base_model,
        training_data=training_data,
        evaluation_data=evaluation_data,
        partition=eval_index,
        format_type='page',
        num_workers=threads,
        resize='both',
    )

    # create training object
    accelerator, device = _device_parser(device)
    trainer = KrakenTrainer(
        devices=device,
        accelerator=accelerator,
        enable_progress_bar=True,
        enable_summary=True,
        val_check_interval=1.0,
        max_epochs=max_epochs,
        min_epochs=min_epochs,
    )

    # start training
    trainer.fit(training_model)

    if training_model.best_epoch == -1:
        click.echo('Did not converge. Exiting...', err=True)
        return
    else:
        click.echo(f'Best model found at epoch: {training_model.best_epoch}')

    best_model_path = training_model.best_model
    shutil.copy(best_model_path, output_path.joinpath(f'{output_name}_best.mlmodel'))
    click.echo(f'Best model saved to: {output_path.joinpath(f"{output_name}_best.mlmodel")}')
