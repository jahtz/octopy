# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import logging
from pathlib import Path

import click

from .util import ClickCallback, parse_device, spinner


logger: logging.Logger = logging.getLogger('octopy')
HIDE: bool = True


@click.command('train')
@click.help_option('--help', hidden=True)
# Data Config
@click.option(
     '-g', '--gt-data', 'training_data',
     help='One or more ground truth PAGE-XML files. Use quotes to enclose glob patterns (e.g., \'*.xml\').',
     type=click.Path(), callback=ClickCallback.expand_glob, multiple=True, required=True
)
@click.option(
     '-e', '--eval-data', 'evaluation_data',
     help='One or more optional evaluation PAGE-XML files. Use quotes to enclose glob patterns (e.g., \'*.xml\').',
     type=click.Path(), callback=ClickCallback.expand_glob, multiple=True
)
@click.option(
     '-t', '--test-data', 'test_data',
     help='One or more optional evaluation PAGE-XML files. Use quotes to enclose glob patterns (e.g., \'*.xml\').',
     type=click.Path(), callback=ClickCallback.expand_glob, multiple=True
)
@click.option(
     '-o', '--output', 'checkpoint_path',
     help='Output directory for saving the model and checkpoints.',
     type=click.Path(file_okay=False, resolve_path=True, path_type=Path), required=True
)
@click.option(
     '-p', '--partition', 'partition',
     help='Split ground truth files into training and evaluation sets if no evaluation files are provided. '
          'Default partition is 90% training, 10% evaluation.',
     type=click.FloatRange(min=0.0, max=1.0), default=0.9, show_default=True
)
@click.option(
     '--num-workers', 'num_workers',
     help='Number of worker processes for CPU-based training.',
     type=click.IntRange(min=1), default=1, show_default=True, hidden=HIDE
)
@click.option(
     '--augment', 'augment',
     help='Switch to enable input image augmentation.',
     type=click.BOOL, is_flag=True
)
@click.option(
     '--data-batch-size', 'data_batch_size',
     help='Number of items to pack into a single sample.',
     type=click.IntRange(min=1), default=1, show_default=True, hidden=HIDE
)
@click.option(
     '--line-position', 'topline',
     help='Indicator of baseline position in dataset.',
     type=click.Choice(['baseline', 'topline', 'centerline']), 
     default='baseline', show_default=True, hidden=HIDE
)
@click.option(
     '-mb', '--merge-lines', 'line_merge',
     help='Baseline merge mapping. One or more mappings of the form \'-mb SOURCE TARGET\', '
          'where \'SOURCE\' is merged into \'TARGET\'.',
     callback=ClickCallback.merge_mapping, multiple=True, nargs=2,
)
@click.option(
     '-mr', '--merge-regions', 'region_merge',
     help='Region merge mapping. One or more mappings of the form \'-mr SOURCE TARGET\', '
          'where \'SOURCE\' is merged into \'TARGET\'.',
     callback=ClickCallback.merge_mapping, multiple=True, nargs=2, 
)
@click.option(
     '-n', '--name', 'name',
     help='Name of the output model. Used for saving results and checkpoints.',
     type=click.STRING, default='model', show_default=False
)
@click.option(
     '--load', 'load',
     help='Load an existing model as a basis for the training process.',
     type=click.Path(dir_okay=False, resolve_path=True, path_type=Path)
)
@click.option(
     '--resume', 'resume',
     help='Resume training from a checkpoint.',
     type=click.Path(dir_okay=False, resolve_path=True, path_type=Path)
)
@click.option(
     '--deterministic', 'deterministic',
     help='Enables deterministic training. If no seed is given and enabled the seed will be set to 42.',
     type=click.BOOL, is_flag=True, hidden=HIDE
)
@click.option(
     '--seed', 'seed',
     help='Number of items to pack into a single sample.',
     type=click.INT, hidden=HIDE
)

# Model config
@click.option(
     '--spec', 'spec',
     help='VGSL spec of the baseline labeling network. See https://kraken.re/5.3.0/vgsl.html for further information.',
     default='[1,1800,0,3 Cr7,7,64,2,2 Gn32 Cr3,3,128,2,2 Gn32 Cr3,3,128 Gn32 Cr3,3,256 Gn32 Cr3,3,256 Gn32 Lbx32 Lby32 Cr1,1,32 Gn32 Lby32 Lbx32]',
     type=click.STRING, show_default=False, hidden=HIDE    
)
@click.option(
     '--padding', 'padding',
     help='Padding (left/right, top/bottom) around the page image.',
     type=click.Tuple([int, int]), default=(0, 0), show_default=True, nargs=2, hidden=HIDE
)
@click.option(
     '--resize',
     help='Controls how the model\'s output layer is resized if the training data contains different classes. '
          '\'union\' adds new classes (former \'add\'), \'new\' resizes to match the training data (former \'both\'), '
          'and \'fail\' aborts training if there is a mismatch.',
     type=click.Choice(['union', 'new', 'fail']), default='new', show_default=True
)
@click.option(
     '--bl-tol', 'bl_tol',
     help='Tolerance in pixels for baseline detection metrics',
     type=click.FloatRange(min=0.0), default=10.0, show_default=True, hidden=HIDE
)
@click.option(
     '--dice-weight', 'dice_weight',
     help='No documentation',
     type=click.FloatRange(min=0.0, max=1.0), default=0.5, show_default=True, hidden=HIDE
)
@click.option(
     '--epochs', 'epochs',
     help='Number of epochs to train for when using fixed stopping.',
     type=click.INT, default=-1, show_default=True
)
@click.option(
     '--completed-epochs', 'completed_epochs',
     help='Number of epochs already completed. Used for resuming training.',
     type=click.IntRange(min=0), default=0, show_default=True, hidden=HIDE
)
@click.option(
     '--freq', 'freq',
     help='Model saving and report generation frequency in epochs during training. '
          'If frequency is >1 it must be an integer, i.e. running validation every n-th epoch.',
     type=click.FLOAT, default=1.0, show_default=True, hidden=HIDE
)
@click.option(
     '-f', '--format', 'weights_format',
     help='Weight format to convert checkpoint at end of training to',
     type=click.Choice(['safetensors', 'coreml']), default='safetensors', show_default=True
)
@click.option(
     '--optimizer', 'optimizer',
     help='Optimizer to use during training.',
     type=click.Choice(['Adam', 'AdamW', 'SGD', 'RMSprop']), default='AdamW', show_default=True, hidden=HIDE
)
@click.option(
     '--lrate', 'lrate',
     help='Learning rate for the optimizer.',
     type=click.FLOAT, default=1e-5, show_default=True, hidden=HIDE
)
@click.option(
     '--momentum', 'momentum',
     help='Momentum parameter for applicable optimizers.',
     type=click.FLOAT, default=0.9, show_default=True, hidden=HIDE
)
@click.option(
     '--weight-decay', 'weight_decay',
     help='Weight decay parameter for the optimizer.',
     type=click.FLOAT, default=0.0, show_default=True, hidden=HIDE
)
@click.option(
     '--gradient-clip-val', 'gradient_clip_val',
     help='Threshold for gradient clipping.',
     type=click.FLOAT, default=1.0, show_default=True, hidden=HIDE
)
@click.option(
     '--accumulate-grad-batches', 'accumulate_grad_batches',
     help='Number of batches to aggregate before backpropagation.',
     type=click.INT, default=1, show_default=True, hidden=HIDE
)
@click.option(
     '--schedule', 'schedule',
     help='Type of learning rate schedule. For 1cycle, cycle length is determined by the `--step-size` option.',
     type=click.Choice(['cosine', 'constant', 'exponential', 'step', '1cycle', 'reduceonplateau']),
     default='constant', show_default=True, hidden=HIDE
)
@click.option(
     '--warmup', 'warmup',
     help='Number of iterations to warmup learning rate.',
     type=click.INT, default=0, show_default=True, hidden=HIDE
)
@click.option(
     '--step-size', 'step_size',
     help='Learning rate decay in stepped schedule.',
     type=click.INT, default=10, show_default=True, hidden=HIDE
)
@click.option(
     '--gamma', 'gamma',
     help='Learning rate decay in exponential schedule.',
     type=click.FLOAT, default=0.1, show_default=True, hidden=HIDE
)
@click.option(
     '--rop-factor', 'rop_factor',
     help='Learning rate decay in reduce on plateau schedule.',
     type=click.FLOAT, default=0.1, show_default=True, hidden=HIDE
)
@click.option(
     '--rop-patience', 'rop_patience',
     help='Number of epochs to wait before reducing learning rate.',
     type=click.INT, default=5, show_default=True, hidden=HIDE
)
@click.option(
     '--cos-t-max', 'cos_t_max',
     help='Epoch at which cosine schedule reaches final learning rate.',
     type=click.INT, default=10, show_default=True, hidden=HIDE
)
@click.option(
     '--cos-min-lr', 'cos_min_lr',
     help='Final learning rate with cosine schedule.',
     type=click.FLOAT, default=1e-6, show_default=True, hidden=HIDE
)
@click.option(
     '-q', '--quit', 'quit',
     help='Stop condition for training. Choose \'early\' for early stopping or \'fixed\' for a fixed number of epochs.',
     type=click.Choice(['early', 'fixed']), default='early', show_default=True
)
@click.option(
     '--min-epochs', 'min_epochs',
     help='Minimum number of epochs to train for before early stopping is allowed.',
     type=click.INT, default=0, show_default=True
)
@click.option(
     '--lag', 'lag',
     help='Early stopping patience (number of validation steps without improvement). Measured by val_mean_iu.',
     type=click.IntRange(min=1), default=10, show_default=True
)
@click.option(
     '--min-delta', 'min_delta',
     help='Minimum delta of validation scores.',
     type=click.FLOAT, default=0.0, show_default=True, hidden=HIDE
)
@click.option(
    '--precision', 'precision',
    help='Numerical precision to use for inference.',
    type=click.Choice(['transformer-engine', 'transformer-engine-float16', '16-true', '16-mixed', 'bf16-true', 'bf16-mixed', '32-true', '64-true']),
    default='32-true', show_default=True, hidden=HIDE
)
@click.option(
     '-d', '--device', 'device',
     help='Specify the device for processing (e.g. cpu, cuda:0, ...). Refer to PyTorch documentation for supported devices.',
     type=click.STRING, default='auto:auto', show_default=True
)
@click.option(
     '--model-batch-size', 'model_batch_size',
     help='Sets the batch size for inference.',
     type=click.IntRange(min=1), default=1, show_default=True, hidden=HIDE
)
@click.option(
     '--num-threads', 'num_threads',
     help='Number of threads to use for intra-op parallelisation.',
     type=click.IntRange(min=1), default=1, show_default=True, hidden=HIDE
)
@click.option(
     '--yes', '-y', 'yes',
     help='Skip training class check.',
     type=click.BOOL, is_flag=True
)
def cli_train(**kwargs) -> None:
     """
     Train a custom segmentation model using Kraken.
     """
     with spinner as sp:
          sp.add_task('Initialize', total=None)
          from octopy.train import Trainer, training_model_config, training_data_config
          
          kwargs['topline'] = {'baseline': False, 'topline': True}.get(kwargs['topline'], None)
          kwargs['accelerator'], kwargs['device'] = parse_device(kwargs['device'])
          
          trainer = Trainer(
               model_config=training_model_config(**kwargs), 
               data_config=training_data_config(**kwargs),
               load=kwargs['load'],
               resume=kwargs['resume'],
               deterministic=kwargs['deterministic'],
               seed=kwargs['seed'],
               console=sp
          )
     
     if not kwargs['yes'] and not click.confirm('Do you want to continue?'):
          return

     trainer.fit(kwargs['name'])
