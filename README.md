# Documentation

# Installation

## PyEnv
GitHub Repository: [PyEnv](https://github.com/pyenv/pyenv)

### Install Python
Recommended Python Version: `3.11.8`
```bash
pyenv install 3.11.8
```
### Create Virtual Environment
Create a PyEnv virtual environment (1)
```bash
pyenv virtualenv 3.11.8 pagesegment
```
or a global venv (2) 
```bash
pyenv local 3.11.8  # select Python 3.11.8
```
```bash
python -m venv /path/to/venv/pagesegment  # create virtual environment
```
```bash
pyenv local system
```

### Activate Virtual Environment
```bash
pyenv activate pagesegment  # (1)
```
or
```bash
source /path/to/venv/pagesegment/bin/activate  # (2)
```

### Install pagesegment
```bash
git clone --recurse-submodules https://github.com/jatzelberger/pagesegment.git
```

### Install dependencies
```bash
pip install -r pagesegment/requirements.txt
```

## Usage
...