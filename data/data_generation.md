# SBA Dataset Refactoring Guide

## Overview

This refactoring consolidates `SBADataset` and `SBADatasetTwo` into a single, flexible class with command-line configuration support.

## Key Changes

### 1. **Unified Dataset Class** (`SBADataset.py`)

**Before:**
- `SBADataset`: Minimal features, stratified split only
- `SBADatasetTwo`: Full features, chronological split, separate leakage method

**After:**
- Single `SBADataset` class with configurable:
  - Feature mode: `minimal` (7 features) or `full` (all features with OHE)
  - Split strategy: `stratified` or `chronological`
  - Train/test/query ratios (default: 60/20/20)
  - Optional data leakage for experiments

### 2. **Three-Way Splits**

All data is now split into **three disjoint sets**:
- **Train**: For model training (default 60%)
- **Test**: For model evaluation (default 20%)
- **Query**: For inference/production simulation (default 20%)

### 3. **Command-Line Interface** (`main.py`)

Added argparse support for easy configuration:

```bash
# Basic usage
python main.py --input data/SBAcase.11.13.17.csv --output-dir data/sba

# Custom splits
python main.py --train-ratio 0.7 --test-ratio 0.15 --query-ratio 0.15

# Different configurations
python main.py --feature-mode minimal --split-strategy stratified
python main.py --feature-mode full --split-strategy chronological

# Presets
python main.py --preset default    # Full features, chronological, 60/20/20
python main.py --preset minimal    # Minimal features, stratified, 60/20/20
python main.py --preset roar       # ROAR config: full, chronological, 80/10/10
python main.py --preset leakage    # Experimental leakage mode
```

## Usage Examples

### 1. Default Configuration (Recommended)

```bash
python main.py --input data/SBAcase.11.13.17.csv
```

This uses:
- Full feature set with one-hot encoding
- Chronological split (temporal validity)
- 60% train, 20% test, 20% query
- Proper scaling (no leakage)

### 2. Minimal Feature Set

```bash
python main.py \
  --feature-mode minimal \
  --split-strategy stratified \
  --output-dir data/sba_minimal
```

Uses only 7 core features: Term, NoEmp, CreateJob, RetainedJob, Portion, RealEstate, RevLineCr

### 3. ROAR Replication

```bash
python main.py \
  --preset roar \
  --input data/SBAcase.11.13.17.csv
```

Replicates ROAR paper configuration:
- Full features
- Chronological split
- 80/10/10 split ratios

### 4. Custom Split Ratios

```bash
python main.py \
  --train-ratio 0.5 \
  --test-ratio 0.25 \
  --query-ratio 0.25
```

### 5. Experimental Leakage Mode

```bash
python main.py --introduce-leakage --output-dir data/sba_leakage
```

⚠️ **Warning:** Only for experiments to study the impact of data leakage!

## Command-Line Arguments

### Input/Output
- `--input`: Path to CSV file (default: `data/SBAcase.11.13.17.csv`)
- `--output-dir`: Save directory (default: `data/sba`)

### Feature Configuration
- `--feature-mode`: `minimal` or `full` (default: `full`)

### Split Configuration
- `--split-strategy`: `stratified` or `chronological` (default: `chronological`)
- `--train-ratio`: Training proportion (default: 0.6)
- `--test-ratio`: Testing proportion (default: 0.2)
- `--query-ratio`: Query proportion (default: 0.2)

### Experimental Options
- `--introduce-leakage`: Scale before split (data leakage)
- `--fold`: Fold number for stratified splits (default: 0)

### Presets
- `--preset`: Use predefined configuration
  - `default`: Full features, chronological, 60/20/20
  - `minimal`: Minimal features, stratified, 60/20/20
  - `roar`: ROAR paper config (80/10/10)
  - `leakage`: Experimental leakage mode

## File Structure

```
project/
├── Dataset.py              # Base dataset class (unchanged)
├── SBADataset.py          # Unified SBA dataset class (NEW)
├── main.py                # CLI script with argparse (UPDATED)
├── npy_to_csv.py          # NPY to CSV converter (UPDATED)
└── data/
    └── sba/               # Default output directory
        ├── sba_train.npy
        ├── sba_test.npy
        ├── sba_query.npy
        └── sba_feature_names.npy
```

## Output Files

Each run generates 4 files:

1. **sba_train.npy**: Training data (N_train × [1 + F])
2. **sba_test.npy**: Testing data (N_test × [1 + F])
3. **sba_query.npy**: Query data (N_query × [1 + F])
4. **sba_feature_names.npy**: Feature names array

Format: First column = target (NoDefault), remaining columns = features

## Converting to CSV

```bash
# Convert default directory
python npy_to_csv.py --data-dir data/sba

# Convert custom directory
python npy_to_csv.py --data-dir data/sba_roar --prefix roar

# Specify output directory
python npy_to_csv.py --data-dir data/sba --output-dir exports/
```

## Migration from Old Code

### Old: Using SBADataset (minimal)
```python
from SBADataset import SBADataset
sba = SBADataset()
X_train, y_train, X_test, y_test = sba.get_data("data/file.csv")
```

### New: Equivalent
```python
from SBADataset import SBADataset
sba = SBADataset(feature_mode="minimal", split_strategy="stratified")
X_train, y_train, X_test, y_test, X_query, y_query = sba.get_data(
    "data/file.csv",
    train_ratio=0.8,
    test_ratio=0.2,
    query_ratio=0.0  # Or use query set
)
```

### Old: Using SBADatasetTwo
```python
from SBADatasetTwo import SBADatasetTwo
sba = SBADatasetTwo()
X_train, y_train, X_test, y_test = sba.get_data("data/file.csv")
```

### New: Equivalent
```python
from SBADataset import SBADataset
sba = SBADataset(feature_mode="full", split_strategy="chronological")
X_train, y_train, X_test, y_test, X_query, y_query = sba.get_data(
    "data/file.csv",
    train_ratio=0.8,
    test_ratio=0.2,
    query_ratio=0.0
)
```

### Old: Leakage mode
```python
sba.get_data_leakage("data/file.csv")
```

### New: Equivalent
```python
sba.get_data("data/file.csv", introduce_leakage=True)
```

## Benefits of Refactoring

1. **Single source of truth**: One class instead of two
2. **Flexibility**: Easy to switch between configurations
3. **Reproducibility**: Command-line args make experiments reproducible
4. **Proper splits**: Three-way disjoint splits (train/test/query)
5. **No code duplication**: Shared logic for both minimal and full modes
6. **Better testing**: Easy to compare leakage vs. no-leakage
7. **User-friendly**: Presets for common configurations

## Backward Compatibility

The old `SBADataset.py` and `SBADatasetTwo.py` files can be kept temporarily for backward compatibility, but new code should use the refactored `SBADataset` class.

## Testing the Refactoring

```bash
# Test all presets
python main.py --preset default
python main.py --preset minimal
python main.py --preset roar
python main.py --preset leakage

# Verify outputs
python npy_to_csv.py --data-dir data/sba
python npy_to_csv.py --data-dir data/sba_minimal
python npy_to_csv.py --data-dir data/sba_roar
python npy_to_csv.py --data-dir data/sba_leakage

# Compare shapes and distributions
```

## Next Steps

1. **Delete old files**: Remove `SBADataset.py` (old version) and `SBADatasetTwo.py`
2. **Update imports**: Change imports in other scripts
3. **Add tests**: Create unit tests for the new class
4. **Documentation**: Update project README with new usage patterns
5. **CI/CD**: Add command-line tests to your pipeline