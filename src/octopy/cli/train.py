# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import logging
from pathlib import Path

import click
from rich.progress import Progress, TextColumn, SpinnerColumn

from .util import parse_device, read_boolean_environment, expand_glob, merge_mapping


logger: logging.Logger = logging.getLogger(__name__)
SHORT_HELP: bool = read_boolean_environment('OCTOPY_EXTENDED_HELP', True)


@click.command('train')
@click.help_option('--help', hidden=SHORT_HELP)

# Data Config
@click.option(
     '-g', '--gt-data', 'training_data',
     help='Ground-truth PAGE-XML files for training (one or more). Glob expressions are supported by wrapping '
          'patterns in quotes (e.g. \'*.xml\')',
     type=click.Path(), 
     callback=expand_glob, 
     multiple=True, 
     required=True
)
@click.option(
     '-e', '--eval-data', 'evaluation_data',
     help='Optional PAGE-XML files for evaluation/validation. If omitted, a validation split is created from the '
          'training set using --partition.',
     type=click.Path(), 
     callback=expand_glob, 
     multiple=True
)
@click.option(
     '-t', '--test-data', 'test_data',
     help='Optional PAGE-XML files for a held-out test set. This is independent from --eval-data and is typically '
          'used for final reporting.',
     type=click.Path(), 
     callback=expand_glob, 
     multiple=True
)
@click.option(
     '-o', '--output', 'checkpoint_path',
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
     '--num-workers',
     help='Number of worker processes for data loading / CPU preprocessing. Increase to improve throughput when input '
          'preparation is the bottleneck.',
     type=click.IntRange(min=1), 
     default=1, 
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
     '--data-batch-size',
     help='Number of training items to pack into a single sample (data-side batching). This is distinct from '
          '--model-batch-size which controls inference batch size.',
     type=click.IntRange(min=1), 
     default=1, 
     show_default=True, 
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
     '-ml', '--merge-lines', 'line_merge',
     help='Merge line classes before training. May be given multiple times as pairs SOURCE TARGET '
          '(e.g. -mb \'heading\' \'text\'). SOURCE labels are remapped into TARGET.',
     callback=merge_mapping, 
     multiple=True, 
     nargs=2,
)
@click.option(
     '-mr', '--merge-regions', 'region_merge',
     help='Merge region classes before training. May be given multiple times as pairs SOURCE TARGET '
          '(e.g. -mr \'caption\' \'text-region\'). SOURCE labels are remapped into TARGET.',
     callback=merge_mapping, 
     multiple=True, 
     nargs=2, 
)
@click.option(
     '-n', '--name',
     help='Base name for the output model and checkpoint files.',
     type=click.STRING, 
     default='model', 
     show_default=False
)
@click.option(
     '--load',
     help='Initialize training from an existing model file (transfer learning / fine-tuning).',
     type=click.Path(dir_okay=False, resolve_path=True, path_type=Path)
)
@click.option(
     '--resume',
     help='Resume training from an existing checkpoint file (restores optimizer state where supported).',
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
     '--seed',
     help='Random seed for training/reproducibility. If omitted, Kraken uses its default seeding behavior '
          '(unless --deterministic is set).',
     type=click.INT, 
     hidden=SHORT_HELP
)

# Model config
@click.option(
     '--spec',
     help='VGSL network spec for the baseline-labeling model. See Kraken\'s VGSL documentation for details '
          '(https://kraken.re/5.3.0/vgsl.html).',
     default='[1,1800,0,3 Cr7,7,64,2,2 Gn32 Cr3,3,128,2,2 Gn32 Cr3,3,128 Gn32 Cr3,3,256 Gn32 Cr3,3,256 Gn32 Lbx32 Lby32 Cr1,1,32 Gn32 Lby32 Lbx32]',
     type=click.STRING, 
     show_default=False, 
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
     '--resize',
     help='How to handle class mismatches between a loaded model and the training data. \'union\' adds new classes to '
          'the output layer, \'new\' resizes to match the training data, and \'fail\' aborts if there is a mismatch.',
     type=click.Choice(['union', 'new', 'fail']), 
     default='new', 
     show_default=True
)
@click.option(
     '--bl-tol',
     help='Tolerance (in pixels) used when computing baseline detection metrics.',
     type=click.FloatRange(min=0.0),
     default=10.0,
     show_default=True,
     hidden=SHORT_HELP
)
@click.option(
     '--dice-weight',
     help='Weight of the Dice loss component (0..1) when using a combined loss. Higher values emphasize region '
          'overlap; lower values emphasize the complementary term.',
     type=click.FloatRange(min=0.0, max=1.0), 
     default=0.5, 
     show_default=True, 
     hidden=SHORT_HELP
)
@click.option(
     '--epochs',
     help='Number of epochs to train for when using fixed stopping (--quit fixed). Use -1 to rely on early stopping.',
     type=click.INT, 
     default=-1, 
     show_default=True
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
     '--freq',
     help='Checkpointing/validation/report frequency in epochs. If greater than 1, validation runs every n-th epoch.',
     type=click.FLOAT, 
     default=1.0, 
     show_default=True, 
     hidden=SHORT_HELP
)
@click.option(
     '-f', '--format', 'weights_format',
     help='Format to export the final model weights to after training completes.',
     type=click.Choice(['safetensors', 'coreml']), 
     default='safetensors', 
     show_default=True
)
@click.option(
     '--optimizer',
     help='Optimizer used during training.',
     type=click.Choice(['Adam', 'AdamW', 'SGD', 'RMSprop']), 
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
     '--gradient-clip-val',
     help='Maximum gradient norm/value before gradients are clipped to stabilize training.',
     type=click.FLOAT, 
     default=1.0, 
     show_default=True, 
     hidden=SHORT_HELP
)
@click.option(
     '--accumulate-grad-batches',
     help='Accumulate gradients over N batches before performing an optimizer step. Useful to simulate larger batch '
          'sizes when memory is limited.',
     type=click.INT, 
     default=1, 
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
     '--warmup',
     help='Number of optimizer steps/iterations to linearly warm up the learning rate.',
     type=click.INT, 
     default=0, 
     show_default=True, 
     hidden=SHORT_HELP
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
     '-q', '--quit',
     help='Stopping strategy: \'early\' uses early stopping, \'fixed\' trains for a fixed number of epochs.',
     type=click.Choice(['early', 'fixed']), 
     default='early', 
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
     '--min-delta',
     help='Minimum improvement required to reset early stopping patience.',
     type=click.FLOAT, 
     default=0.0, 
     show_default=True, 
     hidden=SHORT_HELP
)
@click.option(
     '--precision',
     help='Numeric precision for training/inference. Lower precision can be faster on supported hardware but may '
          'slightly affect convergence.',
     type=click.Choice(['transformer-engine', 'transformer-engine-float16', '16-true', '16-mixed', 'bf16-true', 'bf16-mixed', '32-true', '64-true']),
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
     '--model-batch-size',
     help='Batch size used by the model during training/inference steps.',
     type=click.IntRange(min=1), 
     default=1, 
     show_default=True, 
     hidden=SHORT_HELP
)
@click.option(
     '--num-threads',
     help='Number of threads used for intra-op parallelism.',
     type=click.IntRange(min=1), 
     default=1, 
     show_default=True, 
     hidden=SHORT_HELP
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
          from octopy import Trainer, training_model_config, training_data_config
          
          kwargs['training_data'] = sorted(kwargs['training_data'])
          kwargs['topline'] = {'baseline': False, 'topline': True}.get(kwargs['topline'], None)
          kwargs['accelerator'], kwargs['device'] = parse_device(kwargs['device'])
          kwargs['checkpoint_path'] = kwargs['checkpoint_path'].joinpath('checkpoints').absolute().as_posix()
          
          trainer = Trainer(
               model_config=training_model_config(**kwargs), 
               data_config=training_data_config(**kwargs),
               load=kwargs['load'],
               resume=kwargs['resume'],
               deterministic=kwargs['deterministic'],
               seed=kwargs['seed'],
               console=progress
          )
     
     if not kwargs['yes'] and not click.confirm('Do you want to continue?'):
          return
     trainer.fit(kwargs['name'])
