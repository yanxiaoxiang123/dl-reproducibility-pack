# Project Artifacts

Use this reference when generating reproducible research project files.

## Recommended Layout

```text
project/
  configs/
    default.yaml
  data/
    splits/
      fold_0.json
  src/
    train.py
    evaluate.py
    reproducibility.py
  checkpoints/
  env_lock/
  README.md
  CITATION.cff
  requirements.txt
```

Adapt this to the existing project instead of forcing a new layout.

## Environment Lock

Capture:

- `pip freeze --all` or conda explicit export.
- Python version and executable path.
- PyTorch/TensorFlow version.
- CUDA, cuDNN, driver, and GPU names when available.
- OS, CPU count, and hostname when useful.
- Exact command used for the run.

## README Reproducibility Section

Include:

- Hardware and software versions.
- Dataset source and split file paths.
- Seed values and deterministic settings.
- Training command.
- Evaluation command.
- Resume command.
- Expected metric range or checksum when available.
- Known nondeterministic limitations.

## CITATION.cff Minimum

```yaml
cff-version: 1.2.0
title: "Project Title"
message: "If you use this code, please cite it as below."
authors:
  - family-names: "Family"
    given-names: "Given"
version: "1.0.0"
date-released: "YYYY-MM-DD"
```

## Publication Checklist

- Code commit or release tag is recorded.
- Dataset version and split indices are recorded.
- Hyperparameters are in config files.
- Environment lock is generated.
- Full checkpoints can resume training.
- Results can be reproduced with documented commands.
- Randomness sources and known limitations are stated.
