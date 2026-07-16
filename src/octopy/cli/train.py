# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from dataclasses import fields
import logging
from pathlib import Path

import click
from rich.progress import Progress, TextColumn, SpinnerColumn

from .util import read_boolean_environment, expand_glob, class_merge, class_valid


logger: logging.Logger = logging.getLogger(__name__)
SHORT_HELP: bool = read_boolean_environment('OCTOPY_EXTENDED_HELP', True)


@click.command('train')
@click.help_option('--help', hidden=SHORT_HELP)

# DATA CONFIG
@click.option(
     '-t', '--train-data', 'train_data',
     help='Ground-truth PAGE-XML files for training (one or more). Glob expressions are supported by wrapping '
          'patterns in quotes (e.g. \'*.xml\')',
     type=click.Path(), 
     callback=expand_glob, 
     multiple=True, 
     required=True
)
@click.option(
     '-e', '--eval-data', 'eval_data',
     help='Optional PAGE-XML files for evaluation/validation. If omitted, a validation split is created from the '
          'training set using --partition.',
     type=click.Path(), 
     callback=expand_glob, 
     multiple=True
)
@click.option(
     '-o', '--output', 'output',
     help='Output directory to write checkpoints and the final trained model.',
     type=click.Path(file_okay=False, resolve_path=True, path_type=Path), 
     required=True
)
@click.option(
     '-p', '--partition',
     help='Training/validation split ratio used only when --eval-data is not provided. For example, 0.9 means 90% '
          'training and 10% validation.',
     type=click.FloatRange(min=0.0, max=1.0), 
     default=0.9, 
     show_default=True
)
@click.option(
     '-n', '--name', 'model_name',
     help='Base name for the output model and checkpoint files.',
     type=click.STRING, 
     default='model', 
     show_default=False
)
@click.option(
    '-i', '--image-extension', 'image_ext',
    help='Define a custom image extension. This overwrites the imageFilename attribute.',
    type=click.STRING
)
@click.option(
     '-m', '--model',
     help='Initialize training from an existing model file (transfer learning / fine-tuning).',
     type=click.Path(dir_okay=False, resolve_path=True, path_type=Path)
)
@click.option(
     '--deterministic',
     help='Enable deterministic training. If enabled and --seed is not provided, the seed is set to 42. '
          'Determinism can reduce performance and may not be fully guaranteed across all operations/devices.',
     type=click.BOOL, is_flag=True, 
     hidden=SHORT_HELP
)
@click.option(
     '--line-position', 'topline',
     help='Baseline position convention in the dataset.',
     type=click.Choice(['baseline', 'topline', 'centerline']), 
     default='baseline', 
     show_default=True, 
     hidden=SHORT_HELP
)
@click.option(
    '--no-regions', 'suppress_regions',
    help='Ignore regions in training.',
    type=click.BOOL,
    is_flag=True
)
@click.option(
    '--no-baselines', 'suppress_baselines',
    help='Ignore baselines in training.',
    type=click.BOOL,
    is_flag=True
)
@click.option(
    '-vr', '--valid-regions',
    help='Only train with a subset of defined regions classes, separated by comma. '
         'If not set, train with all regions.',
    callback=class_valid,
)
@click.option(
    '-vb', '--valid-baselines',
    help='Only train with a subset of defined baseline classes, separated by comma. '
         'If not set, train with all regions.',
    callback=class_valid,
)
@click.option(
    '-mr', '--merge-regions',
    help='Merge region classes before training. May be given multiple times as pairs SOURCE,... TARGET '
          '(e.g. \'-mr caption,footer paragraph\'). SOURCE labels are remapped into TARGET.',
    callback=class_merge, 
    multiple=True, 
    nargs=2,
)
@click.option(
    '-mb', '--merge-baselines',
    help='Merge baseline classes before training. May be given multiple times as pairs SOURCE,... TARGET '
          '(e.g. \'-mr default default_new\'). SOURCE labels are remapped into TARGET.',
    callback=class_merge, 
    multiple=True, 
    nargs=2,
)

# TRAINING CONFIG
@click.option(
     '--vgsl',
     help='VGSL network spec for the baseline-labeling model. See Kraken\'s VGSL documentation for details '
          '(https://kraken.re/5.3.0/vgsl.html).',
     default='[1,1800,0,3 Cr7,7,64,2,2 Gn32 Cr3,3,128,2,2 Gn32 Cr3,3,128 Gn32 Cr3,3,256 Gn32 Cr3,3,256 Gn32 Lbx32 '
             'Lby32 Cr1,1,32 Gn32 Lby32 Lbx32]',
     type=click.STRING, 
     show_default=False, 
     hidden=SHORT_HELP    
)
@click.option(
     '--resize',
     help='How to handle class mismatches between a loaded model and the training data. \'union\' adds new classes to '
          'the output layer, \'new\' resizes to match the training data, and \'fail\' aborts if there is a mismatch.',
     type=click.Choice(['union', 'new', 'fail']), 
     default='new', 
     show_default=True
)
@click.option(
     '--frequency',
     help='Checkpointing/validation/report frequency in epochs. If greater than 1, validation runs every n-th epoch.',
     type=click.FLOAT, 
     default=1.0, 
     show_default=True, 
     hidden=SHORT_HELP
)
@click.option(
     '--line-width',
     help='',
     type=click.INT, 
     default=8, 
     show_default=True, 
     hidden=SHORT_HELP
)
@click.option(
     '--padding',
     help='Padding around the page image given as two integers: (left/right, top/bottom).',
     type=click.Tuple([int, int]), 
     default=(0, 0), 
     show_default=True, 
     nargs=2, 
     hidden=SHORT_HELP
)
@click.option(
     '-q', '--quit',
     help='Stopping strategy: \'early\' uses early stopping, \'fixed\' trains for a fixed number of epochs.',
     type=click.Choice(['early', 'fixed']), 
     default='early', 
     show_default=True
)
@click.option(
     '--epochs',
     help='Number of epochs to train for when using fixed stopping (--quit fixed). Use -1 to rely on early stopping.',
     type=click.INT, 
     default=-1, 
     show_default=True
)
@click.option(
     '--min-epochs',
     help='Minimum number of epochs to train before early stopping can trigger.',
     type=click.INT, 
     default=0, 
     show_default=True
)
@click.option(
     '--lag',
     help='Early stopping patience: number of validation checks without improvement before stopping. '
          'Measured against val_mean_iu.',
     type=click.IntRange(min=1), 
     default=10, 
     show_default=True
)
@click.option(
     '--optimizer',
     help='Optimizer used during training.',
     type=click.Choice(['Adam', 'AdamW', 'SGD', 'RMSprop', 'Lamb']), 
     default='AdamW', 
     show_default=True, 
     hidden=SHORT_HELP
)
@click.option(
     '--lrate',
     help='Learning rate for the optimizer.',
     type=click.FLOAT, 
     default=1e-5, 
     show_default=True,
     hidden=SHORT_HELP
)
@click.option(
     '--momentum',
     help='Momentum factor for optimizers that support it (e.g. SGD, RMSprop).',
     type=click.FLOAT, 
     default=0.9, 
     show_default=True, 
     hidden=SHORT_HELP
)
@click.option(
     '--weight-decay',
     help='Weight decay (L2 regularization) applied by some optimizers.',
     type=click.FLOAT, 
     default=0.0, 
     show_default=True, 
     hidden=SHORT_HELP
)
@click.option(
     '--schedule',
     help='Learning rate schedule type. For \'1cycle\', the cycle length is determined by --step-size.',
     type=click.Choice(['cosine', 'constant', 'exponential', 'step', '1cycle', 'reduceonplateau']),
     default='constant', 
     show_default=True, 
     hidden=SHORT_HELP
)
@click.option(
     '--completed-epochs',
     help='Number of epochs already completed (used when resuming training to keep counters consistent).',
     type=click.IntRange(min=0), 
     default=0, 
     show_default=True, 
     hidden=SHORT_HELP
)
@click.option(
     '--augment',
     help='Enable input image augmentation during training.',
     type=click.BOOL, 
     is_flag=True
)
@click.option(
     '--step-size',
     help='Step interval (in epochs) for stepped learning rate decay schedules.',
     type=click.INT, 
     default=10, 
     show_default=True, 
     hidden=SHORT_HELP
)
@click.option(
     '--gamma',
     help='Multiplicative decay factor for exponential learning rate schedules.',
     type=click.FLOAT, 
     default=0.1, 
     show_default=True, 
     hidden=SHORT_HELP
)
@click.option(
     '--rop-factor',
     help='Learning rate reduction factor for ReduceLROnPlateau schedules.',
     type=click.FLOAT, 
     default=0.1, 
     show_default=True, 
     hidden=SHORT_HELP
)
@click.option(
     '--rop-patience',
     help='Patience (in epochs) for ReduceLROnPlateau before reducing the learning rate.',
     type=click.INT, 
     default=5, 
     show_default=True, 
     hidden=SHORT_HELP
)
@click.option(
     '--cos-t-max',
     help='Epoch index where the cosine schedule reaches its minimum learning rate.',
     type=click.INT, 
     default=10, 
     show_default=True, 
     hidden=SHORT_HELP
)
@click.option(
     '--cos-min-lr',
     help='Minimum learning rate reached by the cosine schedule.',
     type=click.FLOAT, 
     default=1e-6, 
     show_default=True, 
     hidden=SHORT_HELP
)
@click.option(
     '--warmup',
     help='Number of optimizer steps/iterations to linearly warm up the learning rate.',
     type=click.INT, 
     default=0, 
     show_default=True, 
     hidden=SHORT_HELP
)

# OTHER CONFIG
@click.option(
     '--seed',
     help='Random seed for training/reproducibility. If omitted, Kraken uses its default seeding behavior '
          '(unless --deterministic is set).',
     type=click.INT, 
     hidden=SHORT_HELP
)
@click.option(
     '--workers',
     help='Number of worker processes for data loading / CPU preprocessing. Increase to improve throughput when input '
          'preparation is the bottleneck.',
     type=click.IntRange(min=1), 
     default=1, 
     show_default=True, 
     hidden=SHORT_HELP
)
@click.option(
     '--threads',
     help='Number of threads used for intra-op parallelism.',
     type=click.IntRange(min=1), 
     default=1, 
     show_default=True, 
     hidden=SHORT_HELP
)
@click.option(
     '--precision',
     help='Numeric precision for training/inference. Lower precision can be faster on supported hardware but may '
          'slightly affect convergence.',
     type=click.Choice(['16', '16-mixed', '32', '32-true', '64', '64-true', 'bf16', 'bf16-mixed']),
     default='32-true', 
     show_default=True, 
     hidden=SHORT_HELP
)
@click.option(
     '-d', '--device',
     help='Compute device specification (e.g. \'auto\', \'cpu\', \'cuda:0\', ...). Refer to PyTorch documentation '
          'for supported values.',
     type=click.STRING, 
     default='auto', 
     show_default=True
)
@click.option(
     '-y', '--yes', 
     help='Start training without promt.',
     type=click.BOOL, 
     is_flag=True
)
def cli_train(**kwargs) -> None:
    """
    Train a Kraken segmentation model from PAGE-XML ground truth.
    """
    with Progress(
        SpinnerColumn(), 
        TextColumn('[progress.description]{task.description}'), 
        transient=True
    ) as progress:
        progress.add_task('Initialize', total=None)
        
        from octopy.train import Trainer, DataConfig, TrainerConfig
        
        kwargs['train_data'] = sorted(kwargs['train_data'])
        kwargs['eval_data'] = sorted(kwargs['eval_data'])

        data_config = DataConfig(
            **{k: v for k, v in kwargs.items() if k in {f.name for f in fields(DataConfig)}}
        )
        trainer_config = TrainerConfig(
            **{k: v for k, v in kwargs.items() if k in {f.name for f in fields(TrainerConfig)}}
        )
        
        trainer = Trainer(
            data_config,
            trainer_config,
            device=kwargs['device'],
            precision=kwargs['precision'],
            threads=kwargs['threads'],
            workers=kwargs['workers'],
            seed=kwargs['seed'],
            console=progress
        )
        
    if not kwargs['yes'] and not click.confirm('Do you want to continue?'):
        return
    trainer.fit()
