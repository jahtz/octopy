# Documentation
Binarize, normalize and segment a set of images using Kraken and Ocropy nlbin.

## Installation
Tested Python version: `3.11.8`

### PyEnv
GitHub Repository: [PyEnv](https://github.com/pyenv/pyenv)

#### Install Python
```bash
pyenv install 3.11.8
```
#### Create Virtual Environment
Create a PyEnv virtual environment (1)
```bash
pyenv virtualenv 3.11.8 pagesegment
```
or a global venv (2) 
```bash
pyenv local 3.11.8  # select Python 3.11.8
python -m venv /path/to/venv/pagesegment  # create virtual environment
pyenv local system  # switch back to system version
```

#### Activate Virtual Environment
```bash
pyenv activate pagesegment  # (1)
```
or
```bash
source /path/to/venv/pagesegment/bin/activate  # (2)
```

#### Download pagesegment
```bash
git clone --recurse-submodules https://github.com/jahtz/pagesegment.git
```

#### Install dependencies
```bash
pip install -r pagesegment/requirements.txt
```

## Usage
```bash
python pagesegment FILES [DIRECTORY] [OPTIONS]
```
- `FILES` (PATH):<br>
Directory containing files or path to single PNG file.
- `DIRECTORY` (PATH):<br>
Directory for processed files. Defaults to FILES directory or parent directory of input file. Filenames of processed files are identical to input files until first dot (e.g. _0001.orig.png_ &#8594; _0001.xml_).
- `-r`, `--regex` (TEXT):<br>
Regex for input FILES selection. Defaults to `*` (select all files).
- `--threads` (INT):<br>
(Not implemented) Set thread count for processing.
- `-B`, `--binarize` (FLAG):<br>
Binarize input images and write output image to DIRECTORY.
- `-b`, `--bin_suffix` (TEXT):<br>
Changes output suffix of binarized files, if -B flag is set. Defaults to `.bin.png`.
- `--threshold` (INT):<br>
Set binarize threshold percentage. Defaults to 50.
- `-N`, `--normalize` (FLAG):<br>
Normalize input images and write output image to DIRECTORY.
- `-n`, `--nrm_suffix` (TEXT):<br>
Changes output suffix of normalized files, if -N flag is set. Defaults to `.nrm.png`.
- `-S`, `--segment` (FLAG):<br>
Segment input images and write output PageXML to DIRECTORY.
- `-s`, `--seg_suffix` (TEXT):<br>
Changes output suffix of PageXML files, if -S flag is set. Defaults to `.xml`.
- `--creator` (TEXT):<br>
Set creator tag in PageXMl metadata.
- `--scale` (INT):<br>
If set, Kraken will recalculate line masks with entered scale factor.
- `--device` (TEXT):<br>
Device to run neural network for segmentation on. Defaults to `cpu`.
- `--model` (PATH):<br>
Path to Kraken model (.mlmodel). **REQUIRED** for segmentation.

## Example
```bash
python pagesegment /path/to/input_dir/ /path/to/output_dir/ -r *.png -BNS --scale 2000 --model /path/to/model.mlmodel
```

## ZPD
Developed at Centre for [Philology and Digitality](https://www.uni-wuerzburg.de/en/zpd/) (ZPD), [University of Würzburg](https://www.uni-wuerzburg.de/en/).
