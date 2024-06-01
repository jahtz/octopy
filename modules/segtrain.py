from pathlib import Path
import shutil
import logging

import click
import numpy as np
from bs4 import BeautifulSoup
from kraken.lib.train import SegmentationModel, KrakenTrainer
from kraken.lib.default_specs import SEGMENTATION_HYPER_PARAMS
from kraken.lib import log


"""
Module: Segmentation Training
Train a segmentation model using Kraken.
"""

__all__ = ['segtrain']


def segtrain(
    ground_truth: list[Path],
    output: Path,
    model_name: str = 'foo',
    model: Path | None = None,
    partition: float = 0.9,
    device: str = 'cpu',
    threads: int = 1,
    max_epochs: int = SEGMENTATION_HYPER_PARAMS['epochs'],
    min_epochs: int = SEGMENTATION_HYPER_PARAMS['min_epochs'],
    quit: str = SEGMENTATION_HYPER_PARAMS['quit'],
    verbosity: int = 0,
    merge_regions: dict[str, str] | None = None,
    yes: bool = True,
):
    """
    Train segmentation model.

    :param ground_truth: list of ground truth files.
    :param output: output directory to save the trained model.
    :param model_name: name of the output model.
    :param model: path to existing model to continue training. If set to None, a new model is trained from scratch.
    :param partition: ground truth data partition ratio between train/validation set.
    :param device: device to run the model on. (see Kraken guide)
    :param threads: number of threads to use (cpu only)
    :param max_epochs: maximum number of epochs to train.
    :param min_epochs: minimum number of epochs to train.
    :param quit: stop condition for training. Set to `early` for early stopping or `fixed` for fixed number of epochs.
    :param verbosity: verbosity level. (0-2)
    :param merge_regions: region merge mapping. One or more mappings of the form `$target:$src` where $src is merged into $target.
    :param yes: skip query.
    """

    def device_parser(d: str) -> tuple:
        auto_devices = ['cpu', 'mps']
        acc_devices = ['cuda', 'tpu', 'hpu', 'ipu']

        if d in auto_devices:
            return d, 'auto'
        elif any([d.startswith(x) for x in acc_devices]):
            dv, i = device.split(':')
            if dv == 'cuda':
                dv = 'gpu'
            return dv, [int(i)]
        else:
            raise Exception(f'Unknown device: {d}')
        
    def analyse_regions(gt: list[Path]):
        regions: dict[str, int] = {}
        for file in gt:
            try:
                with open(file, 'r', encoding='utf-8') as fp:
                    stream = fp.read()
                    bs = BeautifulSoup(stream, 'xml')
            except Exception:
                raise Exception(f'Could not read file: {file}')
            
            for region in bs.find_all('TextRegion'):
                if region.has_attr('type'):
                    _type = region['type']
                elif region.has_attr('custom'):
                    _type = region['custom'].replace('structure {type:', '').replace(';}', '')
                else:
                    _type = 'no-type'
                if _type == '':
                    _type = 'invalid-region'
                regions[_type] = regions.get(_type, 0) + 1
        for key in sorted(regions.keys()):
            click.echo(f'{key}:\t{regions[key]}')

    logging.captureWarnings(True)
    logger = logging.getLogger('kraken')
    log.set_logger(logger, level=30 - min(10 * verbosity, 20))

    # analyse regions
    click.echo('\nFound regions:')
    analyse_regions(ground_truth)
    click.echo()
    if not yes:
        if not input('Start training? [y/n]: ').lower() in ['y', 'yes']:
            return

    # create output directory
    cp_path = output.joinpath('checkpoints')
    cp_path.mkdir(parents=True, exist_ok=True)

    np.random.default_rng(241960353267317949653744176059648850006).shuffle(ground_truth)
    pt = max(1, int(len(ground_truth) * partition))
    training_data = ground_truth[:pt]
    evaluation_data = ground_truth[pt:]
    accelerator, device = device_parser(device)

    accelerator, device = device_parser(device)

    hyper_params = SEGMENTATION_HYPER_PARAMS.copy()
    hyper_params.update({
        'quit': quit,
        'epochs': max_epochs,
        'min_epochs': min_epochs,
    })

    kraken_model = SegmentationModel(
        hyper_params=SEGMENTATION_HYPER_PARAMS,
        output=cp_path.joinpath(model_name).as_posix(),
        model=model,
        format_type='page',
        training_data=training_data,
        evaluation_data=evaluation_data,
        partition=partition,
        num_workers=threads,
        load_hyper_parameters=True,
        resize='both',
        merge_regions=merge_regions,
    )

    trainer = KrakenTrainer(
        accelerator=accelerator,
        devices=device,
        max_epochs=max_epochs if quit == 'fixed' else -1,
        min_epochs=min_epochs,
        enable_progress_bar=True,
        enable_summary=True,
        val_check_interval=1.0,
    )

    # start training
    trainer.fit(kraken_model)

    # check if model improved and save best model
    if kraken_model.best_epoch == -1:
        click.echo('Model did not improve during training. Exiting...', err=True)
        return
    click.echo(f'Best model found at epoch {kraken_model.best_epoch}')
    best_model_path = kraken_model.best_model
    shutil.copy(best_model_path, output.joinpath(f'{model_name}_best.mlmodel'))
    click.echo(f'Saved to: {output.joinpath(f"{model_name}_best.mlmodel")}')
