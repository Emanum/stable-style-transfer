# Setup

```bash
conda create -n modal-playground python=3.11
conda activate modal-playground
pip install modal
```

Afterwards, create a token

```bash
modal token new
```

# Getting Started

```bash
modal run hello-world.py
```

# Stable Diff Demo

```bash
modal run stable-diff-xl.py --prompt "An astronaut riding a green horse"
```