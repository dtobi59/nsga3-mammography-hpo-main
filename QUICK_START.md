# Quick Start Guide

Get started with NSGA-III Mammography HPO in 4 steps.

## Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended) or CPU
- ~10GB disk space for dataset
- ~5GB for outputs

## Step 1: Install Dependencies (5 minutes)

```bash
# Clone repository
git clone https://github.com/dtobi59/nsga3-mammography-hpo.git
cd nsga3-mammography-hpo

# Install requirements
pip install -r requirements.txt
```

## Step 2: Download Dataset (15 minutes)

**Option A: Using Kaggle CLI (Recommended)**

```bash
# Install Kaggle CLI
pip install kaggle

# Setup Kaggle credentials
# 1. Go to https://www.kaggle.com/settings
# 2. Click "Create New API Token"
# 3. Save kaggle.json to ~/.kaggle/

# Download dataset
kaggle datasets download -d shantanughosh/vindr-mammogram-dataset-dicom-to-png

# Extract
unzip vindr-mammogram-dataset-dicom-to-png.zip -d ./kaggle_vindr_data
```

**Option B: Manual Download**

1. Visit: https://www.kaggle.com/datasets/shantanughosh/vindr-mammogram-dataset-dicom-to-png
2. Click "Download"
3. Extract to `./kaggle_vindr_data`

## Step 3: Test Setup (10 minutes)

```bash
# Test 1: Verify setup
python test_kaggle_setup.py
# Should show: ✓ ALL TESTS PASSED!

# Test 2: Train single model
python test_single_config.py --data_root ./kaggle_vindr_data --epochs 2
# Should complete in 5-10 minutes
```

## Step 4: Run Optimization

**Quick Test (30-60 minutes):**
```bash
python test_mini_optimization.py --data_root ./kaggle_vindr_data
```

**Full Optimization (1-2 days):**
1. Edit `example_kaggle_dataset.py`:
   ```python
   DATA_ROOT = "./kaggle_vindr_data"
   OUTPUT_DIR = "./outputs"
   ```

2. Run:
   ```bash
   python example_kaggle_dataset.py
   ```

## Expected Results

After optimization completes, you'll get:

- **Pareto front**: Multiple optimal trade-off solutions
- **Configurations**: Best hyperparameter settings
- **Metrics**: Sensitivity, Specificity, AUC, Model Size
- **Visualizations**: Trade-off plots

Example output:
```
Found 15 Pareto-optimal solutions

Solution 1:
  Backbone: efficientnet_b0
  AUC: 0.8234, Sensitivity: 0.8156, Specificity: 0.7891
  Model Size: 4.01M

Solution 2:
  Backbone: resnet50
  AUC: 0.8312, Sensitivity: 0.8345, Specificity: 0.8123
  Model Size: 23.52M
```

## Common Issues

### "CUDA out of memory"
```bash
# Reduce batch size
python test_single_config.py --data_root ./kaggle_vindr_data --batch_size 8
```

### "Dataset not found"
```bash
# Verify path
ls ./kaggle_vindr_data
# Should show: images/ and vindr_detection_v1_folds.csv

# Rerun setup test
python test_kaggle_setup.py
```

### "No module named X"
```bash
pip install -r requirements.txt
```

## Configuration Options

Adjust these in `example_kaggle_dataset.py` or `config.py`:

```python
# Population size (more = better exploration, slower)
config.nsga3.pop_size = 50  # Default: 50

# Generations (more = better convergence, slower)
config.nsga3.n_generations = 30  # Default: 30

# Epochs per training (more = better accuracy, slower)
hp_config['epochs'] = 20  # Default: varies

# Surrogate ratio (higher = faster, less accurate)
config.nsga3.surrogate_ratio = 0.7  # Default: 0.7 (70% surrogate)
```

## Time Estimates

| Configuration | Pop Size | Generations | Epochs | Time (GPU) |
|--------------|----------|-------------|--------|------------|
| Quick Test   | 6        | 3           | 2      | 30-60 min  |
| Medium       | 20       | 10          | 10     | 6-12 hours |
| Production   | 50       | 30          | 20     | 1-2 days   |
| Research     | 100      | 50          | 30     | 3-7 days   |

## Next Steps

1. **Review results**: Check `outputs/final_results.pkl`
2. **Visualize trade-offs**: See Pareto front plots
3. **Select configuration**: Choose based on your requirements
4. **Retrain**: Use selected config for final model
5. **Deploy**: Export model for clinical use

## Documentation

- `README.md` - Project overview
- `TESTING_GUIDE.md` - Detailed testing instructions
- `example_kaggle_dataset.py` - Complete usage example
- `test_*.py` - Testing scripts

## Getting Help

1. Run tests in order (don't skip!)
2. Check `TESTING_GUIDE.md` for troubleshooting
3. Review error messages carefully
4. Ensure dataset is properly downloaded

## Project Structure

```
hpo-david/
├── config.py              # Configuration
├── dataset.py             # Data loading (supports Kaggle dataset)
├── models.py              # CNN architectures
├── training.py            # Training pipeline
├── optimization.py        # NSGA-III algorithm
├── surrogate.py          # Gaussian Process surrogate
├── test_kaggle_setup.py  # Setup verification
├── test_single_config.py # Single config test
├── test_mini_optimization.py  # Mini optimization
├── example_kaggle_dataset.py  # Full example
└── TESTING_GUIDE.md      # This guide
```

## Tips for Success

1. **Start small**: Always run tests before full optimization
2. **Use GPU**: CPU is 5-10x slower
3. **Monitor progress**: Check checkpoint files
4. **Save outputs**: Results are saved automatically
5. **Adjust settings**: Tune based on your hardware and time constraints

---

**Ready to start?** Run the setup test:
```bash
python test_kaggle_setup.py
```
