# Copyright 2024 Janik Haitz
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This project includes code from the kraken project,
# available at https://github.com/mittagessen/kraken and licensed under
# Apache 2.0 license https://github.com/mittagessen/kraken/blob/main/LICENSE.

from pathlib import Path
from random import shuffle
from typing import Literal, Optional
import logging
import shutil

import click
from PIL import Image
from rich.traceback import install
from threadpoolctl import threadpool_limits
from kraken.lib import log
from kraken.lib.train import SegmentationModel, KrakenTrainer
from kraken.lib.default_specs import SEGMENTATION_HYPER_PARAMS

from modules.util import parse_path_list, parse_path, parse_suffix, expand_path_list, expand_path, validate_merging


def parse_device(d: str) -> tuple:
    """
    Parses the input device string to pytorch accelerator and device strings

    :param d: Encoded device string (see PyTorch documentation).
    :return: tuple containing accelerator string and device integers.
    """
    auto_devices = ['cpu', 'mps']
    acc_devices = ['cuda', 'tpu', 'hpu', 'ipu']
    if d in auto_devices:
        return d, 'auto'
    elif any([d.startswith(x) for x in acc_devices]):
        dv, i = d.split(':')
        if dv == 'cuda':
            dv = 'gpu'
        return dv, [int(i)]
    else:
        raise Exception(f'Unknown device: {d}')
    

def segtrain(ground_truth: list[Path],
             output: Path,
             gt_glob: str = '*.xml',
             evaluation: Optional[Path] = None,
             eval_glob: str = '*.xml',
             partition: float = 0.9,
             model: Optional[Path] = None,
             name: str = 'foo',
             device: str = 'cpu',
             frequency: float = SEGMENTATION_HYPER_PARAMS['freq'],
             quit: Literal['early', 'fixed'] = SEGMENTATION_HYPER_PARAMS['quit'],
             lag: int = SEGMENTATION_HYPER_PARAMS['lag'],
             epochs: int = SEGMENTATION_HYPER_PARAMS['epochs'],
             min_epochs: int = SEGMENTATION_HYPER_PARAMS['min_epochs'],
             resize: Literal['union', 'new', 'fail'] = 'fail',
             workers: int = 1,
             threads: int = 1,
             suppress_regions: bool = False,
             suppress_baselines: bool = False,
             valid_regions: Optional[str] = None,
             valid_baselines: Optional[str] = None,
             merge_regions: Optional[dict[str, str]] = None,
             merge_baselines: Optional[dict[str, str]] = None,
             verbosity: int = 0) -> None:
    """
    Train a segmentation model using Kraken.

    :param ground_truth: List of PageXML files containing ground truth data for training.
        Supports file paths or directories (used with the -g option).
    :param output: Directory where training checkpoints and results will be saved. The directory will be created if
        it does not already exist.
    :param gt_glob: Specify a glob pattern to match PageXML files within directories passed to GROUND_TRUTH.
    :param evaluation: Optional directory containing files for model evaluation.
    :param eval_glob: Specify a glob pattern to match evaluation PageXML files in the --evaluation directory.
    :param partition: If no evaluation directory is provided, this option splits the GROUND_TRUTH files into training
        and evaluation sets. Default split is 90% training, 10% evaluation.
    :param model: Optional model to start from. If not set, training will start from scratch.
    :param name: Name of the output model. Results in filenames such as <name>_best.mlmodel.
    :param device: Select the device to use for training (e.g., `cpu`, `cuda:0`). Refer to PyTorch documentation
        for supported devices.
    :param frequency: Frequency at which to evaluate model on validation set. If frequency is greater than 1, it must
        be an integer, i.e. running validation every n-th epoch.
    :param quit: Stop condition for training. Choose `early` for early stopping or `fixed` for a fixed number of epochs.
    :param lag: For early stopping, the number of validation steps without improvement (measured by val_mean_iu) to
        wait before stopping.
    :param epochs: Number of epochs to train when using fixed stopping.
    :param min_epochs: Minimum number of epochs to train before early stopping is allowed.
    :param resize: Controls how the model\'s output layer is resized if the training data contains different classes.
        `union` adds new classes, `new` resizes to match the training data, and `fail` aborts training if there is
        a mismatch. `new` is recommended.
    :param workers: Number of worker processes for CPU-based training.
    :param threads: Number of threads to use for CPU-based training.
    :param suppress_regions: Disable region segmentation training.
    :param suppress_baselines: Disable baseline segmentation training.
    :param valid_regions: Comma-separated list of valid regions to include in the training. This option is applied
        before region merging.
    :param valid_baselines: Comma-separated list of valid baselines to include in the training.
        This option is applied before baseline merging.
    :param merge_regions: Region merge mapping. Dictionary of mappings of the form `{src: target}`, where `src` is
        merged into `target`.
    :param merge_baselines: Baseline merge mapping. Dictionary of mappings of the form `{src: target}`, where `src` is
        merged into `target`.
    :param verbosity: Set verbosity level for logging. (levels 0-2).
    """
    Image.MAX_IMAGE_PIXELS = 20000 ** 2

    # create logger
    logging.captureWarnings(True)
    logger = logging.getLogger('kraken')
    log.set_logger(logger, level=30 - min(10 * verbosity, 20))
    logging.getLogger("lightning.fabric.utilities.seed").setLevel(logging.ERROR)
    install(suppress=[click])

    ground_truth = expand_path_list(ground_truth, gt_glob)

    # create output and checkpoint directory
    cp_path = output.joinpath('checkpoints')
    cp_path.mkdir(parents=True, exist_ok=True)

    # check parameters and update hyper parameters
    if resize != 'fail' and not model:
        raise click.BadOptionUsage('resize',
                                   'Resize option != `fail` requires loading an existing model')
    if not (0 <= frequency <= 1) and frequency % 1.0 != 0:
        raise click.BadOptionUsage('frequency',
                                   'Frequency needs to be either in the interval [0,1.0] or a positive integer.')

    hyper_params = SEGMENTATION_HYPER_PARAMS.copy()
    hyper_params.update({
        'freq': frequency,
        'quit': quit,
        'epochs': epochs,
        'min_epochs': min_epochs,
        'lag': lag
    })

    if hyper_params['freq'] > 1:
        val_check_interval = {'check_val_every_n_epoch': int(hyper_params['freq'])}
    else:
        val_check_interval = {'val_check_interval': hyper_params['freq']}

    # if no validation set is provided, split ground truth set
    if evaluation:
        training_data = ground_truth
        evaluation_data = expand_path(evaluation, eval_glob)
        partition = 1
    else:
        # method used by escriptorium for evaluation data generation...
        # np.random.default_rng(241960353267317949653744176059648850006).shuffle(ground_truth)
        # partition = max(1, int(len(ground_truth) * (1 - partition)))
        # training_data = ground_truth[partition:]
        # evaluation_data = ground_truth[:partition]
        training_data = ground_truth
        shuffle(training_data)
        evaluation_data = None

    # create computation device parameters
    accelerator, device = parse_device(device)

    # init training
    s_model = SegmentationModel(hyper_params=hyper_params,
                                output=cp_path.joinpath(name).as_posix(),
                                model=model,
                                training_data=training_data,
                                evaluation_data=evaluation_data,
                                partition=partition,  # ignored if evaluation_data is not None.
                                num_workers=workers,
                                load_hyper_parameters=model is not None,  # load only if start model exists.
                                format_type='page',
                                suppress_regions=suppress_regions,
                                suppress_baselines=suppress_baselines,
                                valid_regions=valid_regions if not valid_regions else valid_regions.split(','),
                                valid_baselines=valid_baselines if not valid_baselines else valid_baselines.split(','),
                                merge_regions=merge_regions,
                                merge_baselines=merge_baselines,
                                resize=resize)
    
    # list baseline and region types
    click.echo('Training line types:')
    for k, v in s_model.train_set.dataset.class_mapping['baselines'].items():
        click.echo(f' - {f"{k} ({v})":<30}{s_model.train_set.dataset.class_stats["baselines"][k]:>5}')
    click.echo('Training region types:')
    for k, v in s_model.train_set.dataset.class_mapping['regions'].items():
        click.echo(f' - {f"{k} ({v})":<30}{s_model.train_set.dataset.class_stats["regions"][k]:>5}')
    if not input('Start training? [y/n]: ').lower() in ['y', 'yes']:
        click.echo('Aborted!')
        return
    
    # build lightning trainer
    trainer = KrakenTrainer(accelerator=accelerator,
                            devices=device,
                            precision='32',  # '64', '32', 'bf16', '16'
                            max_epochs=epochs if quit == 'fixed' else -1,
                            min_epochs=min_epochs,
                            enable_progress_bar=True,
                            deterministic=False,
                            **val_check_interval)
    
    # start training
    with threadpool_limits(limits=threads):
        trainer.fit(s_model)

    # check if model improved and save best model
    if s_model.best_epoch == -1:
        click.echo('Model did not improve during training. Exiting...', err=True)
        return
    click.echo(f'Best model found at epoch {s_model.best_epoch}')
    best_model_path = s_model.best_model
    shutil.copy(best_model_path, output.joinpath(f'{name}_best.mlmodel'))
    click.echo(f'Saved to: {output.joinpath(f"{name}_best.mlmodel")}')  


@click.command('segtrain', help="""
    Train a segmentation model using Kraken.

    This tool allows you to train a segmentation model from PageXML files, with options for using pre-trained models, 
    customizing evaluation, and setting advanced training configurations. 

    Images should be stored in the same directory as their corresponding PageXML files.

    GROUND_TRUTH: List of PageXML files containing ground truth data for training. 
    Supports file paths, wildcards, or directories (used with the -g option).
    """)
@click.help_option('--help')
@click.argument('ground_truth',
                type=click.Path(exists=True, file_okay=True, dir_okay=True),
                callback=parse_path_list,
                required=True,
                nargs=-1)
@click.option('-o', '--output',
              help='Directory where training checkpoints and results will be saved. '
                   'The directory will be created if it does not already exist.',
              type=click.Path(exists=False, file_okay=False, dir_okay=True),
              required=True,
              callback=parse_path)
@click.option('-g', '--gt-glob', 'gt_glob',
              help='Specify a glob pattern to match PageXML files within directories passed to GROUND_TRUTH.',
              type=click.STRING,
              default='*.xml',
              show_default=True)
@click.option('-e', '--evaluation',
              help='Optional directory containing files for model evaluation.',
              type=click.Path(exists=True, file_okay=False, dir_okay=True),
              callback=parse_path)
@click.option('-e', '--eval-glob', 'eval_glob',
              help='Specify a glob pattern to match evaluation PageXML files in the --evaluation directory.',
              type=click.STRING,
              callback=parse_suffix,
              default='*.xml',
              show_default=True)
@click.option('-p', '--partition',
              help='If no evaluation directory is provided, this option splits the GROUND_TRUTH files into '
                   'training and evaluation sets. Default split is 90% training, 10% evaluation.',
              type=click.FLOAT,
              default=0.9,
              show_default=True)
@click.option('-m', '--model',
              help='Optional model to start from. If not set, training will start from scratch.',
              type=click.Path(exists=True, file_okay=True, dir_okay=False),
              callback=parse_path)
@click.option('-n', '--name',
              help='Name of the output model. Results in filenames such as <name>_best.mlmodel',
              type=click.STRING,
              default='foo',
              show_default=True)
@click.option('-d', '--device',
              help='Select the device to use for training (e.g., `cpu`, `cuda:0`). '
                   'Refer to PyTorch documentation for supported devices.',
              type=click.STRING,
              default='cpu',
              show_default=True)
@click.option('--frequency',
              help='Frequency at which to evaluate model on validation set. '
                   'If frequency is greater than 1, it must be an integer, i.e. running validation every n-th epoch.',
              type=click.FLOAT,
              default=SEGMENTATION_HYPER_PARAMS['freq'],
              show_default=True)
@click.option('-q', '--quit',
              help='Stop condition for training. Choose `early` for early stopping or `fixed` '
                   'for a fixed number of epochs.',
              type=click.Choice(['early', 'fixed']),
              default=SEGMENTATION_HYPER_PARAMS['quit'],
              show_default=True)
@click.option('-l', '--lag',
              help='For early stopping, the number of validation steps without improvement '
                   '(measured by val_mean_iu) to wait before stopping.',
              type=click.IntRange(min=1),
              default=SEGMENTATION_HYPER_PARAMS['lag'],
              show_default=True)
@click.option('--epochs', 'epochs',
              help='Number of epochs to train when using fixed stopping.',
              type=click.INT,
              default=SEGMENTATION_HYPER_PARAMS['epochs'],
              show_default=True)
@click.option('--min-epochs', 'min_epochs',
              help='Minimum number of epochs to train before early stopping is allowed.',
              type=click.INT,
              default=SEGMENTATION_HYPER_PARAMS['min_epochs'],
              show_default=True)
@click.option('-r', '--resize',
              help='Controls how the model\'s output layer is resized if the training data contains different classes. '
                   '`union` adds new classes, `new` resizes to match the training data, '
                   'and `fail` aborts training if there is a mismatch. `new` is recommended.',
              type=click.Choice(['union', 'new', 'fail']),  # union = add, new = both
              default='fail',
              show_default=True)
@click.option('--workers',
              help='Number of worker processes for CPU-based training.',
              type=click.IntRange(1),
              default=1,
              show_default=True)
@click.option('--threads',
              help='Number of threads to use for CPU-based training.',
              type=click.IntRange(1),
              default=1,
              show_default=True)
@click.option('--suppress-regions', 'suppress_regions',
              help='Disable region segmentation training.',
              type=click.BOOL,
              is_flag=True)
@click.option('--suppress-baselines', 'suppress_baselines',
              help='Disable baseline segmentation training.',
              type=click.BOOL,
              is_flag=True)
@click.option('-vr', '--valid-regions', 'valid_regions',
              help='Comma-separated list of valid regions to include in the training. '
                   'This option is applied before region merging.',
              type=click.STRING)
@click.option('-vb', '--valid-baselines', 'valid_baselines',
              help='Comma-separated list of valid baselines to include in the training. '
                   'This option is applied before baseline merging.',
              type=click.STRING)
@click.option('-mr', '--merge-regions', 'merge_regions',
              show_default=True,
              help='Region merge mapping. One or more mappings of the form `$target:$src`, '
                   'where $src is merged into $target.',
              multiple=True,
              default=None,
              callback=validate_merging)
@click.option('-mb', '--merge-baselines', 'merge_baselines',
              show_default=True,
              help='Baseline merge mapping. One or more mappings of the form `$target:$src`, '
                   'where $src is merged into $target.',
              multiple=True,
              default=None,
              callback=validate_merging)
@click.option('-v', '--verbose', 'verbosity',
              help='Set verbosity level for logging. Use -vv for maximum verbosity (levels 0-2).',
              count=True)
def segtrain_cli(**kwargs):
    segtrain(**kwargs)
