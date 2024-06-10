# Documentation
## Octopy
Command line tool for image manipulation, text segmentation and recognition.<br>
Made for [OCR4all](https://github.com/OCR4all/OCR4all).

## ToDo
- [ ] structure

## Setup (PyEnv)
Recommended Python version: `3.10.14` <br>
Used Kraken version: `4.3.13`

### PyEnv
```shell
curl https://pyenv.run | bash
```
Make sure that [build dependencies](https://github.com/pyenv/pyenv/wiki#suggested-build-environment) are satisfied.<br>
PyEnv [GitHub](https://github.com/pyenv/pyenv)

### Python
```bash
pyenv install 3.10.14
```
#### Virtual Environment
```bash
pyenv virtualenv 3.10.14 octopy  # install
pyenv activate octopy            # activate venv
pyenv deactivate                 # deactivate venv (after use)
```

### Octopy
```
git clone https://github.com/jahtz/octopy.git
```

#### Install requirements
```
pip install -r octopy/requirements.txt
```

## Usage
### Preprocessing
```shell
python octopy pp [OPTIONS] [FILES]...
```
#### Options
```
-h, --help              Show this message and exit.
-o, --output DIRECTORY  Output directory to save the pre-processed files. [required]
-b, --binarize          Binarize images.
-n, --normalize         Normalize images.
-r, --resize            Resize images. Used for binarization and normalization.
--height INTEGER        Height of resized image.
--width INTEGER         Width of resized image. If height and width is set, height is prioritized.
-t, --threshold FLOAT   Threshold percentage for binarization. [default:0.5]
```

### Segmentation
```shell
python octopy seg [OPTIONS] [FILES]...
```
#### Options
```
-h, --help                 Show this message and exit.
-m, --model FILE           Path to segmentation model. [required]
-o, --output DIRECTORY     Output directory to save PageXML files.
-s, --suffix TEXT          Suffix to append to the output file name. e.g. '.seg.xml' results in 'imagename.seg.xml'. [default: .xml]
-d, --device TEXT          Device to run the model on (see Kraken guide). [default: cpu]
-c, --creator TEXT         Creator of the PageXML file.
-r, --recalculate INTEGER  Recalculate line polygons with this factor. Increases compute time significantly.
```

### Segmentation Training
```shell
python octopy segtrain [OPTIONS] [GROUND_TRUTH]...
```
#### Options
```
-h, --help                 Show this message and exit.
-o, --output DIRECTORY     Output directory to save epochs and trained model. [required]
-n, --name TEXT            Name of the output model. [default: foo]
-m, --model FILE           Path to existing model to continue training. If set to None, a new model is trained from scratch.
-p, --partition FLOAT      Ground truth data partition ratio between train/validation set. [default: 0.9]
-d, --device TEXT          Device to run the model on (see Kraken guide). [default: cpu]
-t, --threads INTEGER      Number of threads to use (cpu only). [default: 1]
--max-epochs INTEGER       Maximum number of epochs to train. [default: 50]
--min-epochs INTEGER       Minimum number of epochs to train. [default: 0]
-q, --quit [early|fixed]   Stop condition for training. Set to `early` for early stopping or `fixed` for fixed number of epochs. [default: fixed]
-v, --verbose              Verbosity level. For level 2 use -vv (0-3).
-mr, --merge-regions TEXT  Region merge mapping. One or more mappings of the form `$target:$src` where $src is merged into $target.
```

## Licenses
- **Octopy**: <br>
    This project is using [Apache-2.0](https://github.com/jahtz/octopy/blob/main/LICENSE) Open Source license (see LICENSE file)
- **Kraken**: <br>
    Octopy is using a modified version of [Kraken](https://github.com/mittagessen/kraken) v4.3.13.<br>
    Kraken is licensed under [Apache-2.0](https://github.com/mittagessen/kraken/blob/main/LICENSE) Open Source license.<br>
    Changes:<br>
    - blla.py, segmentation.py: Adding fallback polygon option if polygonizer fails.
- **DUP-ocropy**:<br>
    Octopy is using some methods from [DUP-ocropy](https://github.com/ocropus-archive/DUP-ocropy) for image normalization ([ocropus-nlbin](https://github.com/ocropus-archive/DUP-ocropy/blob/master/ocropus-nlbin)).<br>
    DUP-ocropy is licensed under [Apache-2.0](https://github.com/ocropus-archive/DUP-ocropy/blob/master/LICENSE) Open Source license.<br>
    Changes:<br>
    - ocropus-nlbin.py: changed to work with Python 3.x.

## ZPD
Developed at Centre for [Philology and Digitality](https://www.uni-wuerzburg.de/en/zpd/) (ZPD), [University of Würzburg](https://www.uni-wuerzburg.de/en/).
