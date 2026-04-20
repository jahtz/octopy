# octopy
[![Python Version](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)

CLI toolkit for layout analysis of historical prints using [Kraken](https://github.com/mittagessen/kraken).

## Origin
Since Kraken moved from classic CLI calls to config-based training in v7, I rewrote the CLI to keep automation-friendly “old-style” commands.<br>
It also lets us ship small segmentation fixes as a plugin, without forking Kraken.

## Setup

> [!NOTE]
> The setup process is configured for [uv](https://github.com/astral-sh/uv).

```bash
git clone https://github.com/jahtz/octopy
```

```bash
uv tool install ./octopy --torch-backend <backend>
```
See `$ uv tool install --help` for possible backends.

## Usage

```bash
octopy [OPTIONS] COMMAND
```

```text
$ octopy --help
Usage: octopy [OPTIONS] COMMAND [ARGS]...

  CLI toolkit for layout analysis of historical prints using Kraken

Options:
  --help                          Show this message and exit.
  --version                       Show the version and exit.
  --logging [ERROR|WARNING|INFO]  Set logging level.  [default: ERROR]

Commands:
  inspect  Inspect a segmentation model file and print selected metadata.
  segment  Run Kraken layout analysis (segmentation) on one or more...
  train    Train a Kraken segmentation model from PAGE-XML ground truth.

  Developed at Centre for Philology and Digitality (ZPD), University of
  Würzburg
```

### Inspect
Inspect a segmentation model file and print selected metadata.

```bash
octopy inspect [OPTIONS] MODEL
```

```text
$ octopy inspect --help
Usage: octopy inspect [OPTIONS] MODEL

  Inspect a segmentation model file and print selected metadata.

  MODEL: Path to the segmentation model file to inspect.

Options:
  -a, --all      Print all metadata keys stored in the model file (raw view).
                 Useful for debugging and for discovering available fields.
  -s, --spec     Print the network specification (VGSL) embedded in the model,
                 if present.
  -m, --metrics  Print training metrics stored in the model metadata (e.g.
                 loss/accuracy curves), if present.
```

### Segment
Run Kraken layout analysis (segmentation) on one or more images and write PAGE-XML.

```bash
octopy segment [OPTIONS] IMAGES...
```

> [!TIP]
> To see all options, set environment variable `OCTOPY_VERBOSE_HELP` to `True`:<br>
> `$ export OCTOPY_VERBOSE_HELP="TRUE"`

```text
$ octopy segment --help
Usage: octopy segment [OPTIONS] IMAGES...

  Run Kraken layout analysis (segmentation) on one or more images and write
  PAGE-XML.

  IMAGES: One or more image paths. Glob patterns should be in quotes.

Options:
  -m, --model FILE                Path to a custom Kraken segmentation model
                                  file. If omitted, Kraken's default
                                  segmentation model is used.
  -o, --output DIRECTORY          Output directory for generated PAGE-XML
                                  files. If omitted, each PAGE-XML file is
                                  written next to its input image.
  -d, --device TEXT               Compute device for inference (e.g. 'cpu',
                                  'cuda:0',...). Use 'auto' to let
                                  Kraken/PyTorch choose.  [default: auto]
  -s, --sort                      Sort regions/lines according to the model's
                                  reading-order heuristics after segmentation.
  --suffix TEXT                   Filename suffix (full extension) for
                                  generated PAGE-XML files (e.g. '.xml' or
                                  '.page.xml').  [default: .xml]
  --mode [lines|regions|all]      Segmentation output to generate. The
                                  effective output is limited by what the
                                  selected model provides.  [default: all]
  --direction [horizontal-lr|horizontal-rl|vertical-lr|vertical-rl]
                                  Principal text direction to assume for the
                                  page. This influences reading order and some
                                  post-processing.  [default: horizontal-lr]
  --fallback INTEGER              Fallback polygon height (in pixels) used
                                  when text line polygonization fails. If set,
                                  the tool keeps the baseline and generates a
                                  rectangular line polygon with this height;
                                  if omitted, polygonization failures follow
                                  upstream behavior (the affected line may be
                                  dropped).
```

### Train
Train a Kraken segmentation model from PAGE-XML ground truth.

```bash
octopy train [OPTIONS]
```

> [!TIP]
> To see all options, set environment variable `OCTOPY_VERBOSE_HELP` to `True`:<br>
> `$ export OCTOPY_VERBOSE_HELP="TRUE"`

```text
$ octopy train --help
Usage: octopy train [OPTIONS]

  Train a Kraken segmentation model from PAGE-XML ground truth.

Options:
  -g, --gt-data PATH              Ground-truth PAGE-XML files for training
                                  (one or more). Glob expressions are
                                  supported by wrapping patterns in quotes
                                  (e.g. '*.xml')  [required]
  -e, --eval-data PATH            Optional PAGE-XML files for
                                  evaluation/validation. If omitted, a
                                  validation split is created from the
                                  training set using --partition.
  -t, --test-data PATH            Optional PAGE-XML files for a held-out test
                                  set. This is independent from --eval-data
                                  and is typically used for final reporting.
  -o, --output DIRECTORY          Output directory to write checkpoints and
                                  the final trained model.  [required]
  -p, --partition FLOAT RANGE     Training/validation split ratio used only
                                  when --eval-data is not provided. For
                                  example, 0.9 means 90% training and 10%
                                  validation.  [default: 0.9; 0.0<=x<=1.0]
  --augment                       Enable input image augmentation during
                                  training.
  -ml, --merge-lines TEXT...      Merge line classes before training. May be
                                  given multiple times as pairs SOURCE TARGET
                                  (e.g. -mb 'heading' 'text'). SOURCE labels
                                  are remapped into TARGET.
  -mr, --merge-regions TEXT...    Merge region classes before training. May be
                                  given multiple times as pairs SOURCE TARGET
                                  (e.g. -mr 'caption' 'text-region'). SOURCE
                                  labels are remapped into TARGET.
  -n, --name TEXT                 Base name for the output model and
                                  checkpoint files.
  --load FILE                     Initialize training from an existing model
                                  file (transfer learning / fine-tuning).
  --resume FILE                   Resume training from an existing checkpoint
                                  file (restores optimizer state where
                                  supported).
  --resize [union|new|fail]       How to handle class mismatches between a
                                  loaded model and the training data. 'union'
                                  adds new classes to the output layer, 'new'
                                  resizes to match the training data, and
                                  'fail' aborts if there is a mismatch.
                                  [default: new]
  --epochs INTEGER                Number of epochs to train for when using
                                  fixed stopping (--quit fixed). Use -1 to
                                  rely on early stopping.  [default: -1]
  -f, --format [safetensors|coreml]
                                  Format to export the final model weights to
                                  after training completes.  [default:
                                  safetensors]
  -q, --quit [early|fixed]        Stopping strategy: 'early' uses early
                                  stopping, 'fixed' trains for a fixed number
                                  of epochs.  [default: early]
  --min-epochs INTEGER            Minimum number of epochs to train before
                                  early stopping can trigger.  [default: 0]
  --lag INTEGER RANGE             Early stopping patience: number of
                                  validation checks without improvement before
                                  stopping. Measured against val_mean_iu.
                                  [default: 10; x>=1]
  -d, --device TEXT               Compute device specification (e.g. 'auto',
                                  'cpu', 'cuda:0', ...). Refer to PyTorch
                                  documentation for supported values.
                                  [default: auto]
  -y, --yes                       Start training without promt.
```

## ZPD
Developed at Centre for [Philology and Digitality](https://www.uni-wuerzburg.de/en/zpd/) (ZPD), [University of Würzburg](https://www.uni-wuerzburg.de/en/).
