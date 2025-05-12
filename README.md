# octopy

Command line tool for layout analysis of historical prints using Kraken.

## Setup

### Use available docker image

The current version is available as a prebuild image.

```shell
docker pull ghcr.io/jahtz/octopy:latest
```

```shell
docker run --rm -it --gpus all -v $(pwd):/data ghcr.io/jahtz/octopy:latest [OPTIONS] COMMAND [ARGS]...
```

### Build image from source

1. Clone repository

    ```shell
    git clone https://github.com/jahtz/octopy && cd octopy
    ```

2. Build image

    ```shell
    docker build -t octopy .
    ```

3. Run

    ```shell
    docker run --rm -it --gpus all -v $(pwd):/data octopy [OPTIONS] COMMAND [ARGS]...
    ```

### PIP

>[!NOTE]
> Python: `>=3.11`<br>
> CUDA: `12.5`

>[!TIP]
> Use a virtual enviroment, e.g. with [pyenv](https://github.com/pyenv/pyenv?tab=readme-ov-file#linuxunix).

1. Clone and install custom [Kraken](https://github.com/jahtz/kraken) fork:

    ```shell
    git clone --single-branch --branch octopy https://github.com/jahtz/kraken
    ```

    ```shell
    pip install kraken/.
    ```

2. Clone and install octopy

    ```shell
    git clone https://github.com/jahtz/octopy
    ```

    ```shell
    pip install octopy/.
    ```

3. Set LD\_LIBRARY\_PATH  (for GPU acceleration, optional)

    ```shell
    export LD_LIBRARY_PATH="/usr/local/cuda-12.5/lib64:$LD_LIBRARY_PATH"
    ```

## Usage

```txt
$ octopy --help
Usage: octopy [OPTIONS] COMMAND [ARGS]...

  Command line tool layout analysis and OCR of historical prints using Kraken.

Options:
  --help     Show this message and exit.
  --version  Show the version and exit.

Commands:
  segment   Segment images using Kraken.
  segtrain  Train a custom segmentation model using Kraken.
```

### Layout Segmentation Training

```txt
$ octopy segtrain --help
Usage: octopy segtrain [OPTIONS]

  Train a custom segmentation model using Kraken.

Options:
  -g, --gt FILE                   One or more ground truth PageXML files. Use
                                  quotes to enclose glob patterns (e.g.,
                                  "*.xml").  [required]
  -e, --eval FILE                 One or more optional evaluation PageXML
                                  files. Use quotes to enclose glob patterns
                                  (e.g., "*.xml").
  -p, --partition FLOAT           Split ground truth files into training and
                                  evaluation sets if no evaluation files are
                                  provided. Default partition is 90% training,
                                  10% evaluation.  [default: 0.9]
  -i, --imagesuffix TEXT          Full suffix of the image files to be used.
                                  If not set, the suffix is derived from the
                                  XML files.
  -o, --output DIRECTORY          Output directory for saving the model and
                                  checkpoints.  [required]
  -m, --model FILE                Path to a pre-trained model to fine-tune. If
                                  not set, training starts from scratch.
  -n, --name TEXT                 Name of the output model. Used for saving
                                  results and checkpoints.
  -d, --device TEXT               Specify the device for processing (e.g. cpu,
                                  cuda:0, ...). Refer to PyTorch documentation
                                  for supported devices.  [default: cpu]
  -w, --workers INTEGER RANGE     Number of worker processes for CPU-based
                                  training.  [default: 1; x>=1]
  -t, --threads INTEGER RANGE     Number of threads for CPU-based training.
                                  [default: 1; x>=1]
  -r, --resize [union|new|fail]   Controls how the model's output layer is
                                  resized if the training data contains
                                  different classes. `union` adds new classes
                                  (former `add`), `new` resizes to match the
                                  training data (former `both`), and `fail`
                                  aborts training if there is a mismatch.
                                  [default: new]
  --line-width INTEGER            Height of baselines in the target image
                                  after scaling.  [default: 8]
  --padding <INTEGER INTEGER>...  Padding (left/right, top/bottom) around the
                                  page image.  [default: 0, 0]
  --freq FLOAT                    Model saving and report generation frequency
                                  in epochs during training. If frequency is
                                  >1 it must be an integer, i.e. running
                                  validation every n-th epoch.  [default: 1.0]
  --quit [early|fixed]            Stop condition for training. Choose `early`
                                  for early stopping or `fixed` for a fixed
                                  number of epochs.  [default: fixed]
  --epochs INTEGER                Number of epochs to train for when using
                                  fixed stopping.  [default: 50]
  --min-epochs INTEGER            Minimum number of epochs to train for before
                                  early stopping is allowed.  [default: 0]
  --lag INTEGER RANGE             Early stopping patience (number of
                                  validation steps without improvement).
                                  Measured by val_mean_iu.  [default: 10;
                                  x>=1]
  --optimizer [Adam|SGD|RMSprop|Lamb]
                                  Optimizer to use during training.  [default:
                                  Adam]
  --lrate FLOAT                   Learning rate for the optimizer.  [default:
                                  0.0002]
  --momentum FLOAT                Momentum parameter for applicable
                                  optimizers.  [default: 0.9]
  --weight-decay FLOAT            Weight decay parameter for the optimizer.
                                  [default: 1e-05]
  --schedule [constant|1cycle|exponential|cosine|step|reduceonplateau]
                                  Set learning rate scheduler. For 1cycle,
                                  cycle length is determined by the `--step-
                                  size` option.  [default: constant]
  --completed-epochs INTEGER      Number of epochs already completed. Used for
                                  resuming training.  [default: 0]
  --augment                       Use data augmentation during training.
  --step-size INTEGER             Step size for learning rate scheduler.
                                  [default: 10]
  --gamma FLOAT                   Gamma for learning rate scheduler.
                                  [default: 0.1]
  --rop-factor FLOAT              Factor for reducing learning rate on
                                  plateau.  [default: 0.1]
  --rop-patience INTEGER          Patience for reducing learning rate on
                                  plateau.  [default: 5]
  --cos-t-max INTEGER             Maximum number of epochs for cosine
                                  annealing.  [default: 50]
  --cos-min-lr FLOAT              Minimum learning rate for cosine annealing.
                                  [default: 2e-05]
  --warmup INTEGER                Number of warmup epochs for cosine
                                  annealing.  [default: 0]
  --precision [64|32|bf16|16]     Numerical precision to use for training.
                                  Default is 32-bit single-point precision.
                                  [default: 32]
  --suppress-regions              Disable region segmentation training.
  --suppress-baselines            Disable baseline segmentation training.
  -vr, --valid-regions TEXT       Comma-separated list of valid regions to
                                  include in the training. Applied before
                                  region merging.
  -vb, --valid-baselines TEXT     Comma-separated list of valid baselines to
                                  include in the training. Applied before
                                  baseline merging.
  -mr, --merge-regions TEXT...    Region merge mapping. One or more mappings
                                  of the form '-mr SOURCE TARGET', where
                                  'SOURCE' is merged into 'TARGET'.
  -mb, --merge-baselines TEXT...  Baseline merge mapping. One or more mappings
                                  of the form '-mb SOURCE TARGET', where
                                  'SOURCE' is merged into 'TARGET'.
  --logging [ERROR|WARNING|INFO|DEBUG]
                                  Set logging level.  [default: ERROR]
```

### Layout Segmentation Prediction

```txt
$ octopy segment --help
Usage: octopy segment [OPTIONS] IMAGES...

  Segment images using Kraken.

  IMAGES: Specify one or more image files to segment. Supports multiple file
  paths, wildcards, or directories (with the -g option).

Options:
  -m, --model FILE                Path to custom segmentation model(s). If not
                                  provided, the default Kraken model is used.
  -o, --output DIRECTORY          Output directory for processed files.
                                  Defaults to the parent directory of each
                                  input file.
  -s, --suffix TEXT               Suffix for output PageXML files. Should end
                                  with '.xml'.  [default: .xml]
  -d, --device TEXT               Specify the processing device (e.g. 'cpu',
                                  'cuda:0',...). See PyTorch documentation.
                                  [default: cpu]
  --creator TEXT                  Metadata: Creator of the PageXML files.
                                  [default: octopy]
  --direction [hlr|hrl|vlr|vrl]   Text direction of input images.  [default:
                                  hlr]
  --suppress-lines                Suppress lines in the output PageXML.
  --suppress-regions              Suppress regions in the output PageXML.
                                  Creates a single dummy region for the whole
                                  image.
  --fallback INTEGER              Use a default bounding box when the
                                  polygonizer fails to create a polygon around
                                  a baseline (in pixels).
  --heatmap TEXT                  Generate a heatmap image alongside the
                                  PageXML output. Specify the file extension
                                  for the heatmap (e.g., `.hm.png`).
  --logging [ERROR|WARNING|INFO|DEBUG]
                                  Set logging level.  [default: ERROR]
```

## ZPD
Developed at Centre for [Philology and Digitality](https://www.uni-wuerzburg.de/en/zpd/) (ZPD), [University of Würzburg](https://www.uni-wuerzburg.de/en/).