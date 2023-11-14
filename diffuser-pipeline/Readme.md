# Setup

## Requirements 

Tested on:
- Python 3.10
- Win11
- Nvidia RX3070

Stable diffusion uses pytorch so a cuda enabled GPU and operating system is required.

## Prepare Environment

**Optional Use Miniconda** to create a virtual environment. See [here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) for installation instructions.
```bash
conda create -n stable-style-transfer python=3.10
conda activate table-style-transfer
```

**Install pipenv**
```bash
pip install pipenv
```

**Install dependencies**
```bash
pipenv install
```

**Install PyTorch**

go to 
https://pytorch.org/get-started/locally/
and search for the latest stable version for your system
then install it similar to this
```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

