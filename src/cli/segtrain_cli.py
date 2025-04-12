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

from pathlib import Path

import rich_click as click
from kraken.lib.default_specs import SEGMENTATION_HYPER_PARAMS

from octopy import segtrain
from .util import paths_callback, path_callback, expand_paths, validate_callback, merge_callback, suffix_callback


@click.command("segtrain")
@click.help_option("--help", hidden=True)
@click.option("-g", "--gt", "ground_truth",
              help="Directory containing ground truth XML and matching image files. "
                   "Multiple directories can be specified.",
              type=click.Path(exists=True, dir_okay=True, file_okay=False, resolve_path=True),
              callback=paths_callback, required=True, multiple=True)
@click.option("--gt-glob", "gt_glob",
              help="Glob pattern for matching ground truth XML files within the specified directories.",
              type=click.STRING, default="*.xml", required=False, show_default=True)
@click.option("-e", "--eval", "evaluation",
              help="Optional directory containing evaluation data with matching image files. "
                   "Multiple directories can be specified.",
              type=click.Path(exists=True, dir_okay=True, file_okay=False, resolve_path=True),
              callback=paths_callback, required=False, multiple=True)
@click.option("--eval-glob", "evaluation_glob",
              help="Glob pattern for matching XML files in the evaluation directory.",
              type=click.STRING, default="*.xml", required=False, show_default=True)
@click.option("-p", "--partition", "partition",
              help="Split ground truth files into training and evaluation sets if no evaluation files are provided. "
                   "Default partition is 90% training, 10% evaluation.",
              type=click.FLOAT, default=0.9, show_default=True)
@click.option("-i", "--imagesuffix", "imagesuffix",
              help="Full suffix of the image files to be used. If not set, the suffix is derived from the XML files.",
              type=click.STRING, callback=suffix_callback, required=False)
@click.option("-o", "--output", "output",
              help="Output directory for saving the model and checkpoints.",
              type=click.Path(exists=False, dir_okay=True, file_okay=False, resolve_path=True),
              callback=path_callback, required=True)
@click.option("-m", "--model", "base_model",
              help="Path to a pre-trained model to fine-tune. If not set, training starts from scratch.",
              type=click.Path(exists=True, file_okay=True, dir_okay=False, resolve_path=True),
              callback=path_callback, required=False)
@click.option("-n", "--name", "model_name",
              help="Name of the output model. Used for saving results and checkpoints.",
              type=click.STRING, default="foo", show_default=True, required=False)
@click.option("-d", "--device", "device",
              help="Specify the device for processing (e.g. cpu, cuda:0, ...). "
                   "Refer to PyTorch documentation for supported devices.",
              type=click.STRING, required=False, default="cpu", show_default=True)
@click.option("-w", "--workers", "workers",
              help="Number of worker processes for CPU-based training.",
              type=click.IntRange(min=1), default=1, show_default=True)
@click.option("-t", "--threads", "threads",
              help="Number of threads for CPU-based training.",
              type=click.IntRange(min=1), default=1, show_default=True)
@click.option('-r', '--resize',
              help="Controls how the model's output layer is resized if the training data contains different classes. "
                   "`union` adds new classes (former `add`), `new` resizes to match the training data (former `both`), "
                   "and `fail` aborts training if there is a mismatch.",
              type=click.Choice(["union", "new", "fail"]),  # union = add, new = both
              default="new", show_default=True)
@click.option("--line-width", "line_width",
              help="Height of baselines in the target image after scaling.",
              type=click.INT, default=SEGMENTATION_HYPER_PARAMS['line_width'], show_default=True)
@click.option("--padding", "padding",
              help="Padding (left/right, top/bottom) around the page image.",
              type=click.Tuple([int, int]), default=SEGMENTATION_HYPER_PARAMS['padding'], show_default=True, nargs=2)
@click.option("--freq", "freq",
              help="Model saving and report generation frequency in epochs during training. "
                   "If frequency is >1 it must be an integer, i.e. running validation every n-th epoch.",
              type=click.FLOAT, default=SEGMENTATION_HYPER_PARAMS['freq'], show_default=True)
@click.option("--quit", "quit",
              help="Stop condition for training. Choose `early` for early stopping or `fixed` "
                   "for a fixed number of epochs.",
              type=click.Choice(["early", "fixed"]), default=SEGMENTATION_HYPER_PARAMS["quit"], show_default=True)
@click.option("--epochs", "epochs",
              help="Number of epochs to train for when using fixed stopping.",
              type=click.INT, default=SEGMENTATION_HYPER_PARAMS["epochs"], show_default=True)
@click.option("--min-epochs", "min_epochs",
              help="Minimum number of epochs to train for before early stopping is allowed.",
              type=click.INT, default=SEGMENTATION_HYPER_PARAMS["min_epochs"], show_default=True)
@click.option("--lag", "lag",
              help="Early stopping patience (number of validation steps without improvement). Measured by val_mean_iu.",
              type=click.IntRange(min=1), default=SEGMENTATION_HYPER_PARAMS["lag"], show_default=True)
@click.option("--optimizer", "optimizer",
              help="Optimizer to use during training.",
              type=click.Choice(["Adam", "SGD", "RMSprop", "Lamb"]),
              default=SEGMENTATION_HYPER_PARAMS["optimizer"], show_default=True)
@click.option("--lrate", "lrate",
              help="Learning rate for the optimizer.",
              type=click.FLOAT, default=SEGMENTATION_HYPER_PARAMS["lrate"], show_default=True)
@click.option("--momentum", "momentum",
              help="Momentum parameter for applicable optimizers.",
              type=click.FLOAT, default=SEGMENTATION_HYPER_PARAMS["momentum"], show_default=True)
@click.option("--weight-decay", "weight_decay",
              help="Weight decay parameter for the optimizer.",
              type=click.FLOAT, default=SEGMENTATION_HYPER_PARAMS["weight_decay"], show_default=True)
@click.option("--schedule", "schedule",
              help="Set learning rate scheduler. For 1cycle, cycle length is determined by the `--step-size` option.",
              type=click.Choice(["constant", "1cycle", "exponential", "cosine", "step", "reduceonplateau"]),
              default=SEGMENTATION_HYPER_PARAMS["schedule"], show_default=True)
@click.option("--completed-epochs", "completed_epochs",
              help="Number of epochs already completed. Used for resuming training.",
              type=click.INT, default=SEGMENTATION_HYPER_PARAMS["completed_epochs"], show_default=True)
@click.option("--augment", "augment",
              help="Use data augmentation during training.",
              is_flag=True, default=SEGMENTATION_HYPER_PARAMS["augment"], show_default=True)
@click.option("--step-size", "step_size",
              help="Step size for learning rate scheduler.",
              type=click.INT, default=SEGMENTATION_HYPER_PARAMS["step_size"], show_default=True)
@click.option("--gamma", "gamma",
              help="Gamma for learning rate scheduler.",
              type=click.FLOAT, default=SEGMENTATION_HYPER_PARAMS["gamma"], show_default=True)
@click.option("--rop-factor", "rop_factor",
              help="Factor for reducing learning rate on plateau.",
              type=click.FLOAT, default=SEGMENTATION_HYPER_PARAMS["rop_factor"], show_default=True)
@click.option("--rop-patience", "rop_patience",
              help="Patience for reducing learning rate on plateau.",
              type=click.INT, default=SEGMENTATION_HYPER_PARAMS["rop_patience"], show_default=True)
@click.option("--cos-t-max", "cos_t_max",
              help="Maximum number of epochs for cosine annealing.",
              type=click.INT, default=SEGMENTATION_HYPER_PARAMS["cos_t_max"], show_default=True)
@click.option("--cos-min-lr", "cos_min_lr",
              help="Minimum learning rate for cosine annealing.",
              type=click.FLOAT, default=SEGMENTATION_HYPER_PARAMS["cos_min_lr"], show_default=True)
@click.option("--warmup", "warmup",
              help="Number of warmup epochs for cosine annealing.",
              type=click.INT, default=SEGMENTATION_HYPER_PARAMS["warmup"], show_default=True)
@click.option("--precision", "precision",
              help="Numerical precision to use for training. Default is 32-bit single-point precision.",
              type=click.Choice(['64', '32', 'bf16', '16']), default="32", show_default=True)
@click.option("--suppress-regions", "suppress_regions",
              help="Disable region segmentation training.",
              type=click.BOOL, is_flag=True)
@click.option("--suppress-baselines", "suppress_baselines",
              help="Disable baseline segmentation training.",
              type=click.BOOL, is_flag=True)
@click.option("-vr", "--valid-regions", "valid_regions",
              help="Comma-separated list of valid regions to include in the training. "
                   "This option is applied before region merging.",
              type=click.STRING, callback=validate_callback)
@click.option("-vb", "--valid-baselines", "valid_baselines",
              help="Comma-separated list of valid baselines to include in the training. "
                   "This option is applied before baseline merging.",
              type=click.STRING, callback=validate_callback)
@click.option("-mr", "--merge-regions", "merge_regions",
              help="Region merge mapping. One or more mappings of the form `src:target`, "
                   "where `src` is merged into `target`. `src` can be comma-separated.",
              multiple=True, default=None, callback=merge_callback, show_default=True)
@click.option("-mb", "--merge-baselines", "merge_baselines",
              help="Baseline merge mapping. One or more mappings of the form `src:target`, "
                   "where `src` is merged into `target`. `src` can be comma-separated.",
              multiple=True, default=None, callback=merge_callback, show_default=True)
@click.option("-v", "--verbose", "verbosity",
              help="Set verbosity level for logging. Use -vv for maximum verbosity (levels 0-2).",
              count=True)
def segtrain_cli(ground_truth: list[Path],
                 evaluation: list[Path],
                 gt_glob: str = "*.xml",
                 evaluation_glob: str = "*.xml",
                 **kwargs):
    """
    Train a custom segmentation model using Kraken.
    """
    ground_truth = expand_paths(ground_truth, gt_glob)
    evaluation = expand_paths(evaluation, evaluation_glob)
    segtrain(ground_truth=ground_truth,
             evaluation=evaluation if evaluation else None,
             **kwargs,
             interactive=True)
