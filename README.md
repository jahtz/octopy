# Octopy   
Command line tool layout analysis and OCR of historical prints using Kraken.


## Setup
>[!NOTE]
> Tested Python versions: `3.9.x`-`3.12.x`

>[!IMPORTANT]
>The following setup process uses [PyEnv](https://github.com/pyenv/pyenv?tab=readme-ov-file#linuxunix)

1. Create Virtual Environment
	```shell
	pyenv install 3.12.8
	pyenv virtualenv 3.12.8 octopy
	pyenv activate octopy
	```
2. Clone and install [custom Kraken](https://github.com/jahtz/kraken) version (optional, but recommended)
   ```shell
   git clone --single-branch --branch octopy https://github.com/jahtz/kraken
   pip install kraken/.
   ```
3. Clone repository
	```shell
	git clone https://github.com/jahtz/octopy
	```
4. Install Octopy
	```shell
	pip install octopy/.
	```


## Setup GPU Acceleration
>[!NOTE]
> Tested CUDA versions: `12.5`

```shell
export LD_LIBRARY_PATH="/usr/local/cuda-12.5/lib64:$LD_LIBRARY_PATH"
```


## Usage
### General
```
$ octopy --help
                                                                                          
 Usage: octopy [OPTIONS] COMMAND [ARGS]...                                                
                                                                                          
 Command line tool layout analysis and OCR of historical prints using Kraken.             
                                                                                          
╭─ Options ──────────────────────────────────────────────────────────────────────────────╮
│ --help         Show this message                                                       │
│ --version      Show the version and exit.                                              │
╰────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Commands ─────────────────────────────────────────────────────────────────────────────╮
│ segment        Segment images using Kraken.                                            │
│ segtrain       Train a custom segmentation model using Kraken.                         │
╰────────────────────────────────────────────────────────────────────────────────────────╯
```


### Layout Segmentation Training
```
$ octopy segtrain --help
                                                                                          
 Usage: octopy segtrain [OPTIONS]                                                         
                                                                                          
 Train a custom segmentation model using Kraken.                                          
                                                                                          
╭─ Input ────────────────────────────────────────────────────────────────────────────────╮
│ *  --gt           -g  Directory containing ground truth XML and matching image files.  │
│                       Multiple directories can be specified.                           │
│                       (DIRECTORY)                                                      │
│                       [required]                                                       │
│    --gt-glob          Glob pattern for matching ground truth XML files within the      │
│                       specified directories.                                           │
│                       (TEXT)                                                           │
│                       [default: *.xml]                                                 │
│    --eval         -e  Optional directory containing evaluation data with matching      │
│                       image files. Multiple directories can be specified.              │
│                       (DIRECTORY)                                                      │
│    --eval-glob        Glob pattern for matching XML files in the evaluation directory. │
│                       (TEXT)                                                           │
│                       [default: *.xml]                                                 │
│    --imagesuffix  -i  Full suffix of the image files to be used. If not set, the       │
│                       suffix is derived from the XML files.                            │
│                       (TEXT)                                                           │
│    --partition    -p  Split ground truth files into training and evaluation sets if no │
│                       evaluation files are provided. Default partition is 90%          │
│                       training, 10% evaluation.                                        │
│                       (FLOAT)                                                          │
│                       [default: 0.9]                                                   │
│    --model        -m  Path to a pre-trained model to fine-tune. If not set, training   │
│                       starts from scratch.                                             │
│                       (FILE)                                                           │
╰────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Options ──────────────────────────────────────────────────────────────────────────────╮
│ *  --output              -o   Output directory for saving the model and checkpoints.   │
│                               (DIRECTORY)                                              │
│                               [required]                                               │
│    --name                -n   Name of the output model. Used for saving results and    │
│                               checkpoints.                                             │
│                               (TEXT)                                                   │
│                               [default: foo]                                           │
│    --device              -d   Specify the device for processing (e.g. cpu, cuda:0,     │
│                               ...). Refer to PyTorch documentation for supported       │
│                               devices.                                                 │
│                               (TEXT)                                                   │
│                               [default: cpu]                                           │
│    --workers             -w   Number of worker processes for CPU-based training.       │
│                               (INTEGER RANGE)                                          │
│                               [default: 1; x>=1]                                       │
│    --threads             -t   Number of threads for CPU-based training.                │
│                               (INTEGER RANGE)                                          │
│                               [default: 1; x>=1]                                       │
│    --resize              -r   Controls how the model's output layer is resized if the  │
│                               training data contains different classes. `union` adds   │
│                               new classes (former `add`), `new` resizes to match the   │
│                               training data (former `both`), and `fail` aborts         │
│                               training if there is a mismatch.                         │
│                               (union|new|fail)                                         │
│                               [default: new]                                           │
│    --suppress-regions         Disable region segmentation training.                    │
│    --suppress-baselines       Disable baseline segmentation training.                  │
│    --valid-regions       -vr  Comma-separated list of valid regions to include in the  │
│                               training. This option is applied before region merging.  │
│                               (TEXT)                                                   │
│    --valid-baselines     -vb  Comma-separated list of valid baselines to include in    │
│                               the training. This option is applied before baseline     │
│                               merging.                                                 │
│                               (TEXT)                                                   │
│    --merge-regions       -mr  Region merge mapping. One or more mappings of the form   │
│                               `src:target`, where `src` is merged into `target`. `src` │
│                               can be comma-separated.                                  │
│                               (TEXT)                                                   │
│    --merge-baselines     -mb  Baseline merge mapping. One or more mappings of the form │
│                               `src:target`, where `src` is merged into `target`. `src` │
│                               can be comma-separated.                                  │
│                               (TEXT)                                                   │
│    --verbose             -v   Set verbosity level for logging. Use -vv for maximum     │
│                               verbosity (levels 0-2).                                  │
│                               (INTEGER RANGE)                                          │
╰────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Hyperparameters ──────────────────────────────────────────────────────────────────────╮
│ --line-width          Height of baselines in the target image after scaling. (INTEGER) │
│                       [default: 8]                                                     │
│ --padding             Padding (left/right, top/bottom) around the page image.          │
│                       (<INTEGER INTEGER>...)                                           │
│                       [default: 0, 0]                                                  │
│ --freq                Model saving and report generation frequency in epochs during    │
│                       training. If frequency is >1 it must be an integer, i.e. running │
│                       validation every n-th epoch.                                     │
│                       (FLOAT)                                                          │
│                       [default: 1.0]                                                   │
│ --quit                Stop condition for training. Choose `early` for early stopping   │
│                       or `fixed` for a fixed number of epochs.                         │
│                       (early|fixed)                                                    │
│                       [default: fixed]                                                 │
│ --epochs              Number of epochs to train for when using fixed stopping.         │
│                       (INTEGER)                                                        │
│                       [default: 50]                                                    │
│ --min-epochs          Minimum number of epochs to train for before early stopping is   │
│                       allowed.                                                         │
│                       (INTEGER)                                                        │
│                       [default: 0]                                                     │
│ --lag                 Early stopping patience (number of validation steps without      │
│                       improvement). Measured by val_mean_iu.                           │
│                       (INTEGER RANGE)                                                  │
│                       [default: 10; x>=1]                                              │
│ --optimizer           Optimizer to use during training. (Adam|SGD|RMSprop|Lamb)        │
│                       [default: Adam]                                                  │
│ --lrate               Learning rate for the optimizer. (FLOAT) [default: 0.0002]       │
│ --momentum            Momentum parameter for applicable optimizers. (FLOAT)            │
│                       [default: 0.9]                                                   │
│ --weight-decay        Weight decay parameter for the optimizer. (FLOAT)                │
│                       [default: 1e-05]                                                 │
│ --schedule            Set learning rate scheduler. For 1cycle, cycle length is         │
│                       determined by the `--step-size` option.                          │
│                       (constant|1cycle|exponential|cosine|step|reduceonplateau)        │
│                       [default: constant]                                              │
│ --completed-epochs    Number of epochs already completed. Used for resuming training.  │
│                       (INTEGER)                                                        │
│                       [default: 0]                                                     │
│ --augment             Use data augmentation during training.                           │
│ --step-size           Step size for learning rate scheduler. (INTEGER) [default: 10]   │
│ --gamma               Gamma for learning rate scheduler. (FLOAT) [default: 0.1]        │
│ --rop-factor          Factor for reducing learning rate on plateau. (FLOAT)            │
│                       [default: 0.1]                                                   │
│ --rop-patience        Patience for reducing learning rate on plateau. (INTEGER)        │
│                       [default: 5]                                                     │
│ --cos-t-max           Maximum number of epochs for cosine annealing. (INTEGER)         │
│                       [default: 50]                                                    │
│ --cos-min-lr          Minimum learning rate for cosine annealing. (FLOAT)              │
│                       [default: 2e-05]                                                 │
│ --warmup              Number of warmup epochs for cosine annealing. (INTEGER)          │
│                       [default: 0]                                                     │
│ --precision           Numerical precision to use for training. Default is 32-bit       │
│                       single-point precision.                                          │
│                       (64|32|bf16|16)                                                  │
│                       [default: 32]                                                    │
╰────────────────────────────────────────────────────────────────────────────────────────╯
```


### Layout Segmentation Prediction
```
$ octopy segment --help
                                                                                          
 Usage: octopy segment [OPTIONS] IMAGES...                                                
                                                                                          
 Segment images using Kraken.                                                             
 IMAGES: Specify one or more image files to segment. Supports multiple file paths,        
 wildcards, or directories (with the -g option).                                          
                                                                                          
╭─ Input ────────────────────────────────────────────────────────────────────────────────╮
│ *  IMAGES    (PATH) [required]                                                         │
╰────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Options ──────────────────────────────────────────────────────────────────────────────╮
│ --glob    -g  Glob pattern for matching images in directories. (used with directories  │
│               in IMAGES).                                                              │
│               (TEXT)                                                                   │
│               [default: *.ocropus.bin.png]                                             │
│ --model   -m  Path to custom segmentation model(s). If not provided, the default       │
│               Kraken model is used.                                                    │
│               (FILE)                                                                   │
│ --output  -o  Output directory for processed files. Defaults to the parent directory   │
│               of each input file.                                                      │
│               (DIRECTORY)                                                              │
│ --suffix  -s  Suffix for output PageXML files. Should end with '.xml'. (TEXT)          │
│               [default: .xml]                                                          │
│ --device  -d  Specify the processing device (e.g. 'cpu', 'cuda:0',...). Refer to       │
│               PyTorch documentation for supported devices.                             │
│               (TEXT)                                                                   │
│               [default: cpu]                                                           │
╰────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Fine-Tuning ──────────────────────────────────────────────────────────────────────────╮
│ --creator             Metadata: Creator of the PageXML files. (TEXT) [default: octopy] │
│ --direction           Text direction of input images. (hlr|hrl|vlr|vrl) [default: hlr] │
│ --suppress-lines      Suppress lines in the output PageXML.                            │
│ --suppress-regions    Suppress regions in the output PageXML. Creates a single dummy   │
│                       region for the whole image.                                      │
│ --fallback            Use a default bounding box when the polygonizer fails to create  │
│                       a polygon around a baseline. Requires a box height in pixels.    │
│                       (INTEGER)                                                        │
│ --heatmap             Generate a heatmap image alongside the PageXML output. Specify   │
│                       the file extension for the heatmap (e.g., `.hm.png`).            │
│                       (TEXT)                                                           │
╰────────────────────────────────────────────────────────────────────────────────────────╯
```

## ZPD
Developed at Centre for [Philology and Digitality](https://www.uni-wuerzburg.de/en/zpd/) (ZPD), [University of Würzburg](https://www.uni-wuerzburg.de/en/).