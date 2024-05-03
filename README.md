# Documentation
Binarize, normalize, segment and transcribe images using [Kraken](https://github.com/mittagessen/kraken)
and [DUP-ocropy](https://github.com/ocropus-archive/DUP-ocropy). <br>
Including wrappers for model training.

## Installation
Recommended Python version: `3.10.14` <br>
Used Kraken version: `4.3.13`

### PyEnv
GitHub: [PyEnv](https://github.com/pyenv/pyenv) with [build dependencies](https://github.com/pyenv/pyenv/wiki#suggested-build-environment).
#### Install Python
```bash
pyenv install 3.10.14
```
#### Create virtual environment
```bash
pyenv virtualenv 3.10.14 pagesegment
```

#### Activate virtual environment
```bash
pyenv activate pagesegment
```

### Pagesegment
#### Download
```bash
git clone --recurse-submodules --remote-submodules https://github.com/jahtz/pagesegment.git
```

#### Install dependencies
```bash
pip install -r pagesegment/requirements.txt
```

## Usage
### Baseline Segmentation
```bash
python pagesegment bls FILES [DIRECTORY] [OPTIONS]
```
#### Options
- `-o <directory>`, `--output <directory>` <br>
Output directory. Creates directory if it does not exist.  Default: _FILES directory_
- `-r <regex>`, `--regex <regex>` <br>
Regular expression for selecting input files. Only used when input is a directory. Default: _*.png_
- `-B`, `--bin` <br>
Binarize images. Recommended for segmentation.
- `-b <suffix>`, `--binsuffix <suffix>` <br>
Set output suffix for binarized files. Default: _.bin.png_
- `-N`, `--nrm` <br>
Normalize images. Recommended for recognition.
- `-n <suffix>`, `--nrmsuffix <suffix>` <br>
Set output suffix for normalized files. Default: _.nrm.png_
- `-S`, `--seg` <br>
Segment images and write results to PageXML files.
- `-s <suffix>`, `--segsuffix <suffix>` <br>
Set output suffix for PageXML files. Default: _.xml_
- `-m <path>`, `--model <path>` <br>
Kraken segmentation model path.
- `-d <device>`, `--device <device>` <br>
Set device used for segmentation. Default: _cpu_ <br>
For CUDA: e.g. _cuda:0_
- `--creator <text>` <br>
Set creator attribute of PageXML metadata.
- `--scale <integer>` <br>
Recalculate line polygons after segmentation with factor. Increases compute time significantly. <br>
Good results: _1200_ <br>
Very good results: _2000_
- `--threshold <percentage>` <br>
 Set threshold for image binarization in percent. Default: _50_

#### Example
```bash
python pagesegment bls /path/to/input_dir/ /path/to/output_dir/ -r *.png -BNS --scale 2000 --model /path/to/model.mlmodel
```

### Segmentation Training
```bash
python pagesegment blstrain GT_FILES [OPTIONS]
```
#### Options
- `-gtr <regex>`, `--gtregex <regex>` <br>
Regular expression for selecting ground truth files. Default: _*.xml_
- `-t <directory>`, `--train <directory>` <br>
Additional training files.
- `-tr <regex>`, `--trainregex <regex>` <br>
Regular expression for selecting additional training files. Default: _*.xml_
- `-e <directory>`, `--eval <directory>` <br>
Evaluation files.
- `-tr <regex>`, `--trainregex <regex>` <br>
Regular expression for selecting evaluation files. Default: _*.xml_
- `-p <percentage>`, `--percentage <percentage>` <br>
Percentage of ground truth data used for evaluation. Only used when _-e_ is not set. Default: _10_
- `-d <device>`, `--device <device>` <br>
Set device used for segmentation. Default: _cpu_ <br>
For CUDA: e.g. _cuda:0_
- `-o <directory>`, `--output <directory>` <br>
Output directory. Creates directory if it does not exist.  Default: _FILES directory_
- `-n <name>`, `--name <name>` <br>
Output model name. Results in _\<name\>\_best.mlmodel_. Default: _foo_
- `-m <path>`, `--model <path>` <br>
Base model for training.
- `--threads <integer>` <br>
Number of threads used for computation. Default: _1_
- `--maxepochs <integer>` <br>
Set max training epochs. Default: _50_
- `--minepochs <integer>` <br>
Set min training epochs. Default: _0_ 

#### Example
```bash
python pagesegment blstrain /path/to/gt/ -d cuda:0 -o /path/to/output/directory/ -n col1 --model /path/to/basemodel.mlmodel
```

## TODO
- [x] Add option to binarize images.
- [x] Add option to normalize images.
- [x] Add segmentation option.
- [x] Add segmentation training.
- [ ] Add recognition option.
- [ ] Add recognition training.

## ZPD
Developed at Centre for [Philology and Digitality](https://www.uni-wuerzburg.de/en/zpd/) (ZPD), [University of Würzburg](https://www.uni-wuerzburg.de/en/).
