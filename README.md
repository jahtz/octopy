# Documentation
## Octopy
Command line tool for Kraken text segmentation and recognition. 
Made for [OCR4all](https://github.com/OCR4all/OCR4all).

## Setup
>[!note]
>Tested Python version: `3.11.x`

### Virtual environment (PyEnv)

```shell
pyenv install 3.11.9
pyenv virtualenv 3.11.9 octopy
pyenv activate octopy
```

### Kraken
Clone modified Kraken version
```shell
git clone --single-branch --branch octopy https://github.com/jahtz/kraken
```

Install modified package
```
pip install ./kraken
```

### Octopy
Clone repository
```shell
git clone --recurse-submodules https://github.com/jahtz/octopy
```

## Usage
```shell
python octopy --help
```

### Segtrain
```
> python octopy segtrain --help
Usage: octopy segtrain [OPTIONS] GROUND_TRUTH...

  Train a segmentation model using Kraken.

  This tool allows you to train a segmentation model from PageXML files, with
  options for using pre-trained models,  customizing evaluation, and setting
  advanced training configurations.

  Images should be stored in the same directory as their corresponding PageXML
  files.

  GROUND_TRUTH: List of PageXML files containing ground truth data for
  training.  Supports file paths, wildcards, or directories (used with the -g
  option).

Options:
  --help                         Show this message and exit.
  -o, --output DIRECTORY         Directory where training checkpoints and
                                 results will be saved. The directory will be
                                 created if it does not already exist.
                                 [required]
  -g, --gt-glob TEXT             Specify a glob pattern to match PageXML files
                                 within directories passed to GROUND_TRUTH.
                                 [default: *.xml]
  -e, --evaluation DIRECTORY     Optional directory containing files for model
                                 evaluation.
  -e, --eval-glob TEXT           Specify a glob pattern to match evaluation
                                 PageXML files in the --evaluation directory.
                                 [default: *.xml]
  -p, --partition FLOAT          If no evaluation directory is provided, this
                                 option splits the GROUND_TRUTH files into
                                 training and evaluation sets. Default split
                                 is 90% training, 10% evaluation.  [default:
                                 0.9]
  -m, --model FILE               Optional model to start from. If not set,
                                 training will start from scratch
  -n, --name TEXT                Name of the output model. Results in
                                 filenames such as <name>_best.mlmodel
                                 [default: foo]
  -d, --device TEXT              Select the device to use for training (e.g.,
                                 `cpu`, `cuda:0`). Refer to PyTorch
                                 documentation for supported devices.
                                 [default: cpu]
  --frequency FLOAT              Frequency at which to evaluate model on
                                 validation set. If frequency is greater than
                                 1, it must be an integer, i.e. running
                                 validation every n-th epoch.  [default: 1.0]
  -q, --quit [early|fixed]       Stop condition for training. Choose `early`
                                 for early stopping or `fixed` for a fixed
                                 number of epochs.  [default: fixed]
  -l, --lag INTEGER RANGE        For early stopping, the number of validation
                                 steps without improvement (measured by
                                 val_mean_iu) to wait before stopping.
                                 [default: 10; x>=1]
  --epochs INTEGER               Number of epochs to train when using fixed
                                 stopping.  [default: 50]
  --min-epochs INTEGER           Minimum number of epochs to train before
                                 early stopping is allowed.  [default: 0]
  -r, --resize [union|new|fail]  Controls how the model's output layer is
                                 resized if the training data contains
                                 different classes. `union` adds new classes,
                                 `new` resizes to match the training data, and
                                 `fail` aborts training if there is a
                                 mismatch. `new` is recommended.  [default:
                                 fail]
  --workers INTEGER RANGE        Number of worker processes for CPU-based
                                 training.  [default: 1; x>=1]
  --threads INTEGER RANGE        Number of threads to use for CPU-based
                                 training.  [default: 1; x>=1]
  --suppress-regions             Disable region segmentation training.
  --suppress-baselines           Disable baseline segmentation training.
  -vr, --valid-regions TEXT      Comma-separated list of valid regions to
                                 include in the training. This option is
                                 applied before region merging.
  -vb, --valid-baselines TEXT    Comma-separated list of valid baselines to
                                 include in the training. This option is
                                 applied before baseline merging.
  -mr, --merge-regions TEXT      Region merge mapping. One or more mappings of
                                 the form `$target:$src`, where $src is merged
                                 into $target.
  -mb, --merge-baselines TEXT    Baseline merge mapping. One or more mappings
                                 of the form `$target:$src`, where $src is
                                 merged into $target.
  -v, --verbose                  Set verbosity level for logging. Use -vv for
                                 maximum verbosity (levels 0-2).
```

### segment
```
> python octopy segment --help
Usage: octopy segment [OPTIONS] [IMAGES]...

  Segment images using Kraken and save the results as XML files (PageXML
  format).

  This tool processes one or more images and segments them using a trained
  Kraken model.  The segmented results are saved as XML files, corresponding
  to each input image.

  IMAGES: List of image files to be segmented. Supports multiple file paths,
  wildcards,  or directories (when used with the -g option).

Options:
  --help                          Show this message and exit.
  -g, --glob TEXT                 Specify a glob pattern to match image files
                                  when processing directories in IMAGES.
                                  [default: *]
  -m, --model FILE                Path to a custom segmentation model. If not
                                  provided, the default Kraken model is used.
                                  Multiple models can be specified.
  -o, --output DIRECTORY          Directory to save the output PageXML files.
                                  Defaults to the parent directory of each
                                  input image.
  --text-direction [l2r|r2l|t2b|b2t]
                                  Set the text direction for segmentation.
                                  [default: l2r]
  -s, --suffix TEXT               Append a suffix to the output PageXML file
                                  names. For example, using `.seg.xml` results
                                  in filenames like `imagename.seg.xml`
                                  [default: .xml]
  -c, --creator TEXT              Specify the creator of the PageXML file.
                                  This can be useful for tracking the origin
                                  of segmented files.
  -d, --device TEXT               Specify the device for running the model
                                  (e.g., `cpu`, `cuda:0`). Refer to PyTorch
                                  documentation for supported devices.
                                  [default: cpu]
  --default-polygon <INTEGER INTEGER INTEGER INTEGER>...
                                  If the polygonizer fails to create a polygon
                                  around a baseline, use this option to create
                                  a default polygon instead of discarding the
                                  baseline. The offsets are defined as left,
                                  top, right, and bottom
  --sort-lines                    Sort text lines in each TextRegion based on
                                  their centroids according to the specified
                                  text direction. (Feature not yet
                                  implemented)
  --drop-empty                    Automatically drop empty TextRegions from
                                  the output. (Feature not yet implemented)
```

## Licenses
- **Octopy**: <br>
    This project is using [Apache-2.0](https://github.com/jahtz/octopy/blob/main/LICENSE) Open Source license.
- **Kraken**: <br>
    Octopy is using a modified version of [Kraken](https://github.com/mittagessen/kraken) v5.2.9.<br>
    Kraken is licensed under [Apache-2.0](https://github.com/mittagessen/kraken/blob/main/LICENSE) Open Source license.<br>
    Changes:<br>
    - blla.py, segmentation.py: Adding default_polygon option if polygonizer fails.

## ZPD
Developed at Centre for [Philology and Digitality](https://www.uni-wuerzburg.de/en/zpd/) (ZPD), [University of Würzburg](https://www.uni-wuerzburg.de/en/).
