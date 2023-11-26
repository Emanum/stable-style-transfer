# Cloud Version with Modal.com

## Requirements

- Python 3.10
- Github Account linked to Modal.com (you get a 30 dollar free credits each month)

## Components

There are just two components for the cloud version
* **modal-style-transfer.py** - Which is the actual pipeline that runs the stable diffusion pipeline and hosts a webserver to trigger the style transfer and static HTML files
* **frontend/*** - The frontend files that are served by the webserver. A one-page HTML side with alpinejs and tailwindcss

## Setup

```bash
pip install modal
```

Afterward, create a token

```bash
modal token new
```

Then you can either serve a dev version temporarily or deploy a stable version.

***Dev Version***


```bash
modal serve modal-style-transfer.py
```

***Stable Version***

```bash
modal deploy modal-style-transfer.py
```

# Local Style Transfer

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

