# octopy
[![Python Version](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)

CLI toolkit for layout analysis of historical prints using [Kraken](https://github.com/mittagessen/kraken).

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
  --help         Show this message and exit.
  --version      Show the version and exit.
  -v, --verbose  Set the verbosity level. Use -v for WARNING, -vv for INFO,
                 -vvv for DEBUG. [default: ERROR]

Commands:
  inspect  Inspect a segmentation model file and print selected metadata.
  predict  Run Kraken layout analysis (segmentation) on one or more...
  train    Train a Kraken segmentation model from PAGE-XML ground truth.
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

### Predict
Run Kraken layout analysis (segmentation) on one or more images and write PAGE-XML.

```bash
octopy predict [OPTIONS] IMAGES...
```

> [!TIP]
> To see all options, set environment variable `OCTOPY_VERBOSE_HELP` to `True`:<br>
> `$ export OCTOPY_VERBOSE_HELP="TRUE"`

```text
$ octopy predict --help
Usage: octopy predict [OPTIONS] IMAGES...

  Run Kraken layout analysis (segmentation) on one or more images and write
  PAGE-XML.

  IMAGES: One or more image paths. Glob patterns should be in quotes.

Options:
  -m, --model FILE               Path to a custom Kraken segmentation model
                                 file. If omitted, Kraken's default
                                 segmentation model is used.
  -o, --output DIRECTORY         Output directory for generated PAGE-XML
                                 files. If omitted, each PAGE-XML file is
                                 written next to its input image.
  -d, --device TEXT              Compute device for inference (e.g. 'cpu',
                                 'cuda:0',...).  [default: cpu]
  -s, --sort                     Sort regions/lines according to the model's
                                 reading-order heuristics after segmentation.
  --suffix TEXT                  Filename suffix (full extension) for
                                 generated PAGE-XML files (e.g. '.xml' or
                                 '.page.xml').  [default: .xml]
  --mode [lines|regions|all]     Segmentation output to generate. The
                                 effective output is limited by what the
                                 selected model provides.  [default: all]
  --polygonizer [kraken|octopy]  Set the type of polygonizer used for baseline
                                 segmentation. 'kraken' uses the default
                                 polygonizer, 'octopy' follows the original
                                 behavior with minor fixes and additions.
                                 [default: octopy]
  --line-fallback INTEGER        Fallback bounding box height (in pixels) used
                                 when text line polygonization fails. Requires
                                 '--polygonizer' to be set to 'octopy'.
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
  -t, --train-data PATH           Ground-truth PAGE-XML files for training
                                  (one or more). Glob expressions are
                                  supported by wrapping patterns in quotes
                                  (e.g. '*.xml')  [required]
  -e, --eval-data PATH            Optional PAGE-XML files for
                                  evaluation/validation. If omitted, a
                                  validation split is created from the
                                  training set using --partition.
  -o, --output DIRECTORY          Output directory to write checkpoints and
                                  the final trained model.  [required]
  -p, --partition FLOAT RANGE     Training/validation split ratio used only
                                  when --eval-data is not provided. For
                                  example, 0.9 means 90% training and 10%
                                  validation.  [default: 0.9; 0.0<=x<=1.0]
  -n, --name TEXT                 Base name for the output model and
                                  checkpoint files.
  -i, --image-extension TEXT      Define a custom image extension. This
                                  overwrites the imageFilename attribute.
  -m, --model FILE                Initialize training from an existing model
                                  file (transfer learning / fine-tuning).
  --no-regions                    Ignore regions in training.
  --no-baselines                  Ignore baselines in training.
  -vr, --valid-regions TEXT       Only train with a subset of defined regions
                                  classes, separated by comma. If not set,
                                  train with all regions.
  -vb, --valid-baselines TEXT     Only train with a subset of defined baseline
                                  classes, separated by comma. If not set,
                                  train with all regions.
  -mr, --merge-regions TEXT...    Merge region classes before training. May be
                                  given multiple times as pairs SOURCE,...
                                  TARGET (e.g. '-mr caption,footer
                                  paragraph'). SOURCE labels are remapped into
                                  TARGET.
  -mb, --merge-baselines TEXT...  Merge baseline classes before training. May
                                  be given multiple times as pairs SOURCE,...
                                  TARGET (e.g. '-mr default default_new').
                                  SOURCE labels are remapped into TARGET.
  --resize [union|new|fail]       How to handle class mismatches between a
                                  loaded model and the training data. 'union'
                                  adds new classes to the output layer, 'new'
                                  resizes to match the training data, and
                                  'fail' aborts if there is a mismatch.
                                  [default: new]
  -q, --quit [early|fixed]        Stopping strategy: 'early' uses early
                                  stopping, 'fixed' trains for a fixed number
                                  of epochs.  [default: early]
  --epochs INTEGER                Number of epochs to train for when using
                                  fixed stopping (--quit fixed). Use -1 to
                                  rely on early stopping.  [default: -1]
  --min-epochs INTEGER            Minimum number of epochs to train before
                                  early stopping can trigger.  [default: 0]
  --lag INTEGER RANGE             Early stopping patience: number of
                                  validation checks without improvement before
                                  stopping. Measured against val_mean_iu.
                                  [default: 10; x>=1]
  --augment                       Enable input image augmentation during
                                  training.
  -d, --device TEXT               Compute device specification (e.g. 'auto',
                                  'cpu', 'cuda:0', ...). Refer to PyTorch
                                  documentation for supported values.
                                  [default: auto]
  -y, --yes                       Start training without prompt.
```

## ZPD
Developed at Centre for [Philology and Digitality](https://www.uni-wuerzburg.de/en/zpd/) (ZPD), [University of Würzburg](https://www.uni-wuerzburg.de/en/).
