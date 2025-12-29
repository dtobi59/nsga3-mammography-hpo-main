# Testing Guide

Complete guide for testing the NSGA-III Mammography HPO project.

## Quick Start

```bash
# 1. Setup test (2 min)
python test_kaggle_setup.py

# 2. Single config test (5-10 min)
python test_single_config.py --data_root /path/to/data

# 3. Mini optimization test (30-60 min)
python test_mini_optimization.py --data_root /path/to/data
```

## Detailed Testing Steps

### 1. Environment Setup Test

**Purpose**: Verify all dependencies and dataset are correctly installed.

```bash
python test_kaggle_setup.py
```

**What it checks**:
- ✓ Required Python packages (PyTorch, pandas, pymoo, etc.)
- ✓ Dataset path exists
- ✓ CSV metadata file found
- ✓ PNG images accessible
- ✓ Dataset class initialization
- ✓ CSV processing and label extraction
- ✓ Image path resolution
- ✓ Train/val split creation
- ✓ Image loading pipeline
- ✓ Model creation and forward pass

**Expected output**:
```
✓ ALL TESTS PASSED!
```

**Common issues**:
- Missing packages → Run `pip install -r requirements.txt`
- Dataset not found → Check path and download dataset
- No PNG files → Verify dataset extraction

---

### 2. Single Configuration Test

**Purpose**: Test complete training loop with one hyperparameter configuration.

```bash
python test_single_config.py --data_root /path/to/data --epochs 2
```

**Options**:
- `--data_root`: Path to Kaggle VinDr dataset (required)
- `--epochs`: Number of training epochs (default: 2)
- `--batch_size`: Batch size (default: 16)
- `--device`: cuda or cpu (default: cuda)

**What it tests**:
- Dataset loading
- Data augmentation
- Model training
- Evaluation metrics
- GPU/CPU computation

**Expected time**: 5-10 minutes (2 epochs)

**Expected output**:
```
RESULTS
Performance Metrics:
  Sensitivity:  0.xxxx
  Specificity:  0.xxxx
  AUC:          0.xxxx
  Accuracy:     0.xxxx

Model Characteristics:
  Model Size:   x.xxM parameters

✓ TEST COMPLETED SUCCESSFULLY!
```

**Common issues**:
- CUDA out of memory → Reduce `--batch_size` to 8 or use `--device cpu`
- Slow on CPU → Use Google Colab with GPU
- Low metrics → Normal for 2 epochs, increase `--epochs`

---

### 3. Mini Optimization Test

**Purpose**: Run small NSGA-III optimization to verify full pipeline.

```bash
python test_mini_optimization.py --data_root /path/to/data
```

**Options**:
- `--data_root`: Path to Kaggle VinDr dataset (required)
- `--output_dir`: Output directory (default: ./test_output)
- `--pop_size`: Population size (default: 6)
- `--generations`: Number of generations (default: 3)
- `--epochs`: Epochs per training (default: 2)
- `--device`: cuda or cpu (default: cuda)

**What it tests**:
- NSGA-III algorithm
- Surrogate model training
- Multi-objective optimization
- Pareto front generation
- Result saving and checkpointing

**Expected time**: 30-60 minutes

**Expected output**:
```
OPTIMIZATION COMPLETE!

Statistics:
  Pareto solutions found: x
  True evaluations: xx
  Surrogate evaluations: xx

Top 3 solutions (by AUC):
  Solution 1:
    AUC: 0.xxxx
    ...

✓ MINI OPTIMIZATION TEST PASSED!
```

**Output files**:
- `test_output/final_results.pkl` - Complete results
- `test_output/checkpoints/` - Generation checkpoints
- `test_output/pareto_front_mini.png` - Visualization

**Common issues**:
- Takes too long → Reduce `--pop_size` to 4
- Memory issues → Reduce `--batch_size` in code
- Errors after first gen → Check checkpoint files

---

### 4. Full Optimization

**Purpose**: Production hyperparameter optimization.

**Setup**:
1. Edit `example_kaggle_dataset.py`
2. Set `DATA_ROOT` to your dataset path
3. Set `OUTPUT_DIR` to your output directory
4. Adjust configuration:
   ```python
   config.nsga3.pop_size = 50         # Population size
   config.nsga3.n_generations = 30    # Generations
   config.nsga3.surrogate_ratio = 0.7 # Surrogate usage
   ```

**Run**:
```bash
python example_kaggle_dataset.py
```

**Expected time**: Hours to days depending on settings

**Recommended settings**:

| Setting | Quick Test | Production | Heavy Research |
|---------|-----------|------------|----------------|
| pop_size | 6-10 | 50 | 100 |
| n_generations | 3-5 | 30 | 50+ |
| epochs | 2-5 | 20 | 30-50 |
| Time | 1-2 hours | 1-2 days | 3-7 days |

---

## Troubleshooting

### CUDA Out of Memory

**Symptoms**: RuntimeError: CUDA out of memory

**Solutions**:
1. Reduce batch size:
   ```python
   test_config['batch_size'] = 8  # Instead of 16
   ```

2. Use CPU (slower):
   ```bash
   python test_single_config.py --device cpu
   ```

3. Use Google Colab with GPU (free)

### Dataset Not Loading

**Symptoms**: FileNotFoundError or "No valid images found"

**Solutions**:
1. Run setup test to diagnose:
   ```bash
   python test_kaggle_setup.py
   ```

2. Verify dataset structure:
   ```
   kaggle_vindr_data/
   ├── images/
   │   └── *.png
   └── vindr_detection_v1_folds.csv
   ```

3. Check CSV column names match expectations

### Slow Training

**Symptoms**: Each epoch takes >10 minutes

**Solutions**:
1. Use PNG dataset (not DICOM) - 5-10x faster
2. Use GPU instead of CPU
3. Reduce image size (edit in training.py)
4. Use fewer samples for testing

### Import Errors

**Symptoms**: ModuleNotFoundError

**Solutions**:
```bash
pip install -r requirements.txt
```

Or install individually:
```bash
pip install torch torchvision
pip install pymoo scikit-learn pandas numpy
pip install pydicom opencv-python albumentations
pip install matplotlib tqdm
```

### Poor Performance Metrics

**Symptoms**: AUC < 0.6, Low sensitivity/specificity

**Solutions**:
1. Increase epochs (2 → 10+)
2. Not a bug - models need more training
3. Check class imbalance in dataset
4. Verify labels are correct

---

## Testing Checklist

Before running full optimization:

- [ ] Setup test passes (`test_kaggle_setup.py`)
- [ ] Single config test completes (`test_single_config.py`)
- [ ] Mini optimization runs successfully (`test_mini_optimization.py`)
- [ ] GPU available (check with `torch.cuda.is_available()`)
- [ ] Sufficient disk space for outputs (>10GB)
- [ ] Dataset fully downloaded and extracted
- [ ] Can access results directory

---

## Performance Benchmarks

Expected performance on different hardware:

### Single Training Run (1 config, 2 epochs)
- GPU (T4/V100): 3-5 minutes
- GPU (RTX 3080): 2-3 minutes
- CPU (8 cores): 15-30 minutes

### Mini Optimization (6 pop, 3 gen, 2 epochs)
- GPU (T4/V100): 30-45 minutes
- GPU (RTX 3080): 20-30 minutes
- CPU: Not recommended (2-4 hours)

### Full Optimization (50 pop, 30 gen, 20 epochs)
- GPU (T4/V100): 24-48 hours
- GPU (RTX 3080): 12-24 hours
- Multiple GPUs: 6-12 hours (with parallelization)

---

## Next Steps After Testing

Once all tests pass:

1. **Adjust configuration** for your needs
2. **Run full optimization** with production settings
3. **Analyze results** using visualization scripts
4. **Select best configuration** from Pareto front
5. **Retrain final model** with more epochs
6. **Evaluate on test set** (not used in optimization)

---

## Support

If issues persist:
1. Check error messages carefully
2. Review this guide's troubleshooting section
3. Run tests in order (don't skip steps)
4. Check GitHub issues for similar problems
