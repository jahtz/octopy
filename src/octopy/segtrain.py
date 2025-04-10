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

import logging
from pathlib import Path
from shutil import copy
from typing import Optional, Literal

from PIL import Image
import rich_click as click
from rich.progress import SpinnerColumn, TextColumn, Progress
from rich import print as rprint
from rich.traceback import install
from threadpoolctl import threadpool_limits
from kraken.lib import log
from kraken.lib.train import SegmentationModel, KrakenTrainer
from kraken.lib.default_specs import SEGMENTATION_HYPER_PARAMS

from .util import device_parser


Image.MAX_IMAGE_PIXELS = 20000 ** 2


def segtrain(ground_truth: list[Path],
             output: Path,
             evaluation: Optional[list[Path]] = None,
             partition: float = 0.9,
             base_model: Optional[Path] = None,
             model_name: str = "foo",
             device: str = "cpu",
             workers: int = 1,
             threads: int = 1,
             resize: Literal['union', 'new', 'fail'] = "new",
             line_width: int = SEGMENTATION_HYPER_PARAMS["line_width"],
             padding: tuple[int, int] = SEGMENTATION_HYPER_PARAMS["padding"],
             freq: float = SEGMENTATION_HYPER_PARAMS["freq"],
             quit: Literal["early", "fixed"] = SEGMENTATION_HYPER_PARAMS["quit"],
             epochs: int = SEGMENTATION_HYPER_PARAMS["epochs"],
             min_epochs: int = SEGMENTATION_HYPER_PARAMS["min_epochs"],
             lag: int = SEGMENTATION_HYPER_PARAMS["lag"],
             optimizer: Literal["Adam", "SGD", "RMSprop", "Lamb"] = SEGMENTATION_HYPER_PARAMS["optimizer"],
             lrate: float = SEGMENTATION_HYPER_PARAMS["lrate"],
             momentum: float = SEGMENTATION_HYPER_PARAMS["momentum"],
             weight_decay: float = SEGMENTATION_HYPER_PARAMS["weight_decay"],
             schedule: Literal["constant", "1cycle", "exponential", "cosine", "step", "reduceonplateau"] = SEGMENTATION_HYPER_PARAMS["schedule"],
             completed_epochs: int = SEGMENTATION_HYPER_PARAMS["completed_epochs"],
             augment: bool = SEGMENTATION_HYPER_PARAMS["augment"],
             step_size: int = SEGMENTATION_HYPER_PARAMS["step_size"],
             gamma: float = SEGMENTATION_HYPER_PARAMS["gamma"],
             rop_factor: float = SEGMENTATION_HYPER_PARAMS["rop_factor"],
             rop_patience: int = SEGMENTATION_HYPER_PARAMS["rop_patience"],
             cos_t_max: int = SEGMENTATION_HYPER_PARAMS["cos_t_max"],
             cos_min_lr: float = SEGMENTATION_HYPER_PARAMS["cos_min_lr"],
             warmup: int = SEGMENTATION_HYPER_PARAMS["warmup"],
             precision: Literal['64', '32', 'bf16', '16'] = "32",
             suppress_regions: bool = False,
             suppress_baselines: bool = False,
             valid_regions: Optional[list[str]] = None,
             valid_baselines: Optional[list[str]] = None,
             merge_regions: Optional[dict[str, str]] = None,
             merge_baselines: Optional[dict[str, str]] = None,
             verbosity: int = 0,
             interactive: bool = False):
    """
    Train a custom segmentation model using Kraken.
    Args:
        ground_truth: List of ground truth PageXML files.
        output: Output directory for saving the model and checkpoints.
        evaluation: Optional list of PageXML files for evaluation.
        partition: Split ground truth files into training and evaluation sets if no evaluation files are provided.
        base_model: Path to a pre-trained model to fine-tune. If not set, training starts from scratch.
        model_name: Name of the output model. Used for saving results and checkpoints.
        device: Specify the device for processing (e.g. cpu, cuda:0, ...). Refer to PyTorch documentation
            for supported devices.
        workers: Number of worker processes for CPU-based training.
        threads: Number of threads for CPU-based training.
        resize: Controls how the model's output layer is resized if the training data contains different classes.
            `union` adds new classes (former `add`), `new` resizes to match the training data (former `both`),
            and `fail` aborts training if there is a mismatch.
        line_width: Height of baselines in the target image after scaling.
        padding: Padding (left/right, top/bottom) around the page image.
        freq: Model saving and report generation frequency in epochs during training.
            If frequency is >1 it must be an integer, i.e. running validation every n-th epoch.
        quit: Stop condition for training. Choose `early` for early stopping or `fixed` for a fixed number of epochs.
        epochs: Number of epochs to train for when using fixed stopping.
        min_epochs: Minimum number of epochs to train for before early stopping is allowed.
        lag: Early stopping patience (number of validation steps without improvement). Measured by val_mean_iu.
        optimizer: Optimizer to use during training.
        lrate: Learning rate for the optimizer.
        momentum: Momentum parameter for applicable optimizers.
        weight_decay: Weight decay parameter for the optimizer.
        schedule: Set learning rate scheduler. For 1cycle, cycle length is determined by the `--step-size` option.
        completed_epochs: Number of epochs already completed. Used for resuming training.
        augment: Use data augmentation during training.
        step_size: Step size for learning rate scheduler.
        gamma: Gamma for learning rate scheduler.
        rop_factor: Factor for reducing learning rate on plateau.
        rop_patience: Patience for reducing learning rate on plateau.
        cos_t_max: Maximum number of epochs for cosine annealing.
        cos_min_lr: Minimum learning rate for cosine annealing.
        warmup: Number of warmup epochs for cosine annealing.
        precision: Numerical precision to use for training. Default is 32-bit single-point precision.
        suppress_regions: Disable region segmentation training.
        suppress_baselines: Disable baseline segmentation training.
        valid_regions: List of regions to include in the training. Use all region if set to None.
        valid_baselines: List of baselines to include in the training. Use all baselines if set to None.
        merge_regions: Dictionary for region merging.
        merge_baselines: Dictionary for baseline merging.
        verbosity: Set verbosity level (0-2).
        interactive: Enable interactive mode for training.
    """
    # TODO: add min_delta to hyperparams and precision options
    # create logger
    logging.captureWarnings(True)
    logger = logging.getLogger("kraken")
    log.set_logger(logger, level=30 - min(10 * verbosity, 20))
    logging.getLogger("lightning.fabric.utilities.seed").setLevel(logging.ERROR)
    install(suppress=[click])

    # create output directory
    cp_path = output.joinpath('checkpoints')
    cp_path.mkdir(parents=True, exist_ok=True)

    # check and update hyperparameters
    if resize != "fail" and not base_model:
        raise click.BadParameter(f"Resize option != `fail` requires loading an existing model.")
    if not (0 <= freq <= 1) and freq % 1.0 != 0:
        raise click.BadParameter(f"Frequency needs to be either in the interval [0,1.0] or a positive integer.")
    hyper_params = SEGMENTATION_HYPER_PARAMS.copy()
    hyper_params.update({
        "line_width": line_width,
        "padding": padding,
        "freq": freq,
        "quit": quit,
        "epochs": epochs,
        "min_epochs": min_epochs,
        "lag": lag,
        "optimizer": optimizer,
        "lrate": lrate,
        "momentum": momentum,
        "weight_decay": weight_decay,
        "schedule": schedule,
        "completed_epochs": completed_epochs,
        "augment": augment,
        "step_size": step_size,
        "gamma": gamma,
        "rop_factor": rop_factor,
        "rop_patience": rop_patience,
        "cos_t_max": cos_t_max,
        "cos_min_lr": cos_min_lr,
        "warmup": warmup,
    })
    if hyper_params["freq"] > 1:
        val_check_interval = {"check_val_every_n_epoch": int(hyper_params["freq"])}
    else:
        val_check_interval = {"val_check_interval": float(hyper_params["freq"])}

    # parse computation device
    accelerator, device = device_parser(device)

    # TODO: add spinner for loading files
    # initialize training
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), transient=True) as spinner:
        spinner.add_task(description="Initialize training", total=None)
        segmentation_model = SegmentationModel(hyper_params=hyper_params,
                                               output=cp_path.joinpath(model_name).as_posix(),
                                               model=base_model,
                                               training_data=ground_truth,
                                               evaluation_data=evaluation,
                                               partition=1 if evaluation else partition,  # ignored if evaluation_data is not None.
                                               num_workers=workers,
                                               load_hyper_parameters=base_model is not None,  # load only if start model exists.
                                               format_type="page",
                                               suppress_regions=suppress_regions,
                                               suppress_baselines=suppress_baselines,
                                               valid_regions=None if not valid_regions else valid_regions,
                                               valid_baselines=None if not valid_baselines else valid_baselines,
                                               merge_regions=merge_regions,
                                               merge_baselines=merge_baselines,
                                               resize=resize)

    # print file summary
    rprint("[bold]Found Files:[/bold]")
    rprint(f" - Training:           {len(segmentation_model.train_set):>5}")
    rprint(f" - Validation:         {len(segmentation_model.val_set):>5}")

    # list baseline and region types
    rprint("[bold]Region Types:[/bold]")
    for k, v in segmentation_model.train_set.dataset.class_mapping["regions"].items():
        rprint(f" - {f'{k}':<20}{segmentation_model.train_set.dataset.class_stats['regions'][k]:>5}")
    rprint("[bold]Baseline Types:[/bold]")
    for k, v in segmentation_model.train_set.dataset.class_mapping["baselines"].items():
        rprint(f" - {f'{k}':<20}{segmentation_model.train_set.dataset.class_stats['baselines'][k]:>5}")
    if interactive:
        print()
        if not input("Start training? [y/n]: ").lower() in ['y', "yes"]:
            rprint("[red]Aborted![/red]")
            return

    # build lightning trainer
    print()  # add empty line for better readability
    kraken_trainer = KrakenTrainer(accelerator=accelerator,
                                   devices=device,
                                   precision=precision,
                                   max_epochs=epochs if quit == "fixed" else -1,
                                   min_epochs=min_epochs,
                                   enable_progress_bar=True,
                                   deterministic=False,
                                   **val_check_interval)

    # start training
    with threadpool_limits(limits=threads):
        kraken_trainer.fit(segmentation_model)

    # check if model improved and save best model
    if segmentation_model.best_epoch == -1:
        # TODO: why no colored output?
        rprint("[orange]INFO:[/orange] Model did not improve during training. Exiting...")
        return
    rprint(f"Best model found at epoch {segmentation_model.best_epoch} with metric {segmentation_model.best_metric}")
    best_model_path = segmentation_model.best_model
    copy(best_model_path, output.joinpath(f"{model_name}_best.mlmodel"))
    rprint(f"Saved to: {output.joinpath(f'{model_name}_best.mlmodel')}")
