# dl-reproducibility-pack

A deep learning reproducibility toolkit for PyTorch and TensorFlow researchers.

## Overview

This skill helps researchers make their deep learning projects fully reproducible for paper publication. It provides:

- Environment templates (requirements.txt, Dockerfile, environment.yml)
- Deterministic training code for PyTorch and TensorFlow
- Standard project structure for research projects
- Configuration management system
- Publication-ready documentation templates

## Installation

### Option 1: Install via Claude Code

```
/plugin marketplace add <this-skill>
```

Or tell Claude Code:

```
我需要你装一个skill，地址是 <owner>/dl-reproducibility-pack，你帮我安装上。
```

### Option 2: Manual Installation

Clone or copy this skill to your Claude Code skills directory:

```bash
# Global installation
~/.claude/skills/dl-reproducibility-pack/

# Or project-local
your-project/.claude/skills/dl-reproducibility-pack/
```

## Usage

### Activating the Skill

Once installed, Claude Code will automatically activate this skill when you:

- Work on PyTorch or TensorFlow research projects
- Ask about reproducibility, random seeds, or determinism
- Need to create environment files or documentation

### Manual Activation

Tell Claude Code:

```
Please use the dl-reproducibility-pack skill to make my project reproducible.
```

## Components

### SKILL.md

The main skill definition containing all templates and best practices for:
- Environment setup (requirements.txt, Dockerfile, environment.yml)
- Random seed setting for PyTorch and TensorFlow
- Project structure
- Configuration management
- README and CITATION.cff templates

### scripts/

Supporting Python utilities:

- `reproducibility.py` - Core seed-setting functions
- `seed_worker.py` - DataLoader worker seeding
- `config.py` - Configuration management

## Features

### 1. Environment & Dependencies

- requirements.txt with pinned versions
- environment.yml for conda
- Dockerfile for complete containerization

### 2. Random Seeds & Determinism

- PyTorch: `set_seed(seed, framework="pytorch", deterministic=True)`
- TensorFlow: `set_seed(seed, framework="tensorflow")`
- DataLoader worker seeding

### 3. Project Structure

```
research_project/
├── configs/
├── data/
├── src/
├── logs/
├── scripts/
├── tests/
├── environment.yml
├── requirements.txt
├── Dockerfile
├── README.md
└── CITATION.cff
```

### 4. Configuration Management

YAML-based hyperparameter management with type-safe Python config classes.

## Example Usage

### Make a Project Reproducible

```bash
# Tell Claude:
"我需要你用 dl-reproducibility-pack 帮我把项目改成可复现的"
```

### Use Seed Setting

```python
from src.reproducibility import set_seed

# PyTorch
set_seed(42, framework="pytorch", deterministic=True)

# TensorFlow
set_seed(42, framework="tensorflow")
```

### Use Reproducible DataLoader

```python
from src.seed_worker import create_reproducible_dataloader

train_loader = create_reproducible_dataloader(
    train_dataset,
    batch_size=128,
    seed=42,
    num_workers=4
)
```

### Use Configuration

```python
from src.config import load_config

config = load_config("configs/default.yaml")
print(config.training.batch_size)  # 128
```

## License

MIT License

## Contributing

Contributions welcome! Please submit issues and pull requests on GitHub.
