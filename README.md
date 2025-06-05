# CIL2D: Class Incremental Learning with Drift Detection and Data Augmentation for Dynamic Processes

This repository contains the implementation of Class Incremental Learning with Drift Detection and Data Augmentation (CIL2D). CIL2D is designed to handle dynamic business processes through a combination of drift detection, data augmentation, and incremental learning techniques.

## Overview

CIL2D addresses the challenge of next activity prediction in dynamic business processes where:
- New activities may appear over time
- Activity orders may change (concept drift)

The approach combines:
- Drift detection using prototype-based distance metrics
- Data augmentation for handling drifting activities
- Incremental learning with replay buffer to prevent catastrophic forgetting

## Getting Started

### Prerequisites

- Python 3.7+
- PyTorch
- pandas
- numpy
- scikit-learn
- matplotlib

### Virtual Environment

Create a virtual environment and activate it:

```bash
python -m venv venv
source venv/bin/activate
```

### Installation

Install the required dependencies in venv:

```bash
pip install -r requirements.txt
```

## Usage

### Step 1: Data Processing

First, process the raw event logs to create prefixes for training and testing:

```bash
python data_processing.py --dataset Sepsis
```

You can specify a different dataset using the `--dataset` parameter. The script will:
- Load the raw event log
- Create prefixes for next activity prediction
- Save the processed data to `./data/{dataset}/processed/prefixes.csv`

### Step 2: Run CIL2D

After processing the data, run the CIL2D algorithm for incremental learning and prediction:

```bash
python CIL2D.py --dataset Sepsis
```

This will:
1. Load the processed data
2. Train an initial model on the training set
3. Process test batches incrementally
4. Detect concept drift
5. Generate augmented samples for drifting activities
6. Update the model
For advanced usage, see the full parameter list in `CIL2D.py`.

## Project Structure

```
.
├── CIL2D.py                  # Main algorithm implementation
├── data_processing.py        # Data preprocessing script
├── lib/
│   ├── data/
│   │   ├── IncrementalDataLoader.py  # Data loading utilities
│   │   └── processor.py     # Log processing utilities
│   └── model/
│       └── incremental_model.py  # Model architecture and training functions
├── data/                     # Data directory
└── results/                  # Results directory
```

## License

LGPL-3.0 license 