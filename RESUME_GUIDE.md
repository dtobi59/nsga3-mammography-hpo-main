# Resume Optimization Guide

This guide explains how to pause and resume optimization runs.

## Overview

The optimization automatically saves checkpoints every 5 generations to your Google Drive. If your Colab session times out or disconnects, you can resume exactly where you left off.

## How It Works

### Automatic Checkpointing

- **Location**: `/content/drive/MyDrive/nsga3_outputs/checkpoints/`
- **Frequency**: Every 5 generations (configurable in `config.nsga3.checkpoint_frequency`)
- **Files**: `checkpoint_gen_5.pkl`, `checkpoint_gen_10.pkl`, etc.

### What's Saved in Each Checkpoint

- Full population state (all solutions)
- Objective values for all solutions
- Surrogate model training data
- Evaluation counts (true and surrogate)
- Optimization history
- Current generation number

## How to Resume

### Method 1: Auto-Resume (Recommended)

Use `resume_from='auto'` to automatically find and resume from the latest checkpoint:

```python
from optimization import run_optimization

results = run_optimization(
    hp_space=config.hyperparameter_space,
    nsga_config=config.nsga3,
    eval_function=eval_function,
    output_dir=OUTPUT_DIR,
    seed=RANDOM_SEED,
    resume_from='auto'  # ← Finds latest checkpoint automatically
)
```

**Output when resuming:**
```
======================================================================
RESUMING FROM CHECKPOINT
======================================================================
Checkpoint: /content/drive/MyDrive/nsga3_outputs/checkpoints/checkpoint_gen_10.pkl
Generation: 10
Previous evals: 200 (True=150, Surrogate=50)
======================================================================

Resuming NSGA-III: Completed 10/30 generations
Running 20 more generations...
```

### Method 2: Resume from Specific Checkpoint

Specify the exact checkpoint file to resume from:

```python
results = run_optimization(
    hp_space=config.hyperparameter_space,
    nsga_config=config.nsga3,
    eval_function=eval_function,
    output_dir=OUTPUT_DIR,
    seed=RANDOM_SEED,
    resume_from='/content/drive/MyDrive/nsga3_outputs/checkpoints/checkpoint_gen_15.pkl'
)
```

## Common Scenarios

### Scenario 1: Colab Timed Out

Your Colab session disconnected after 5 hours:

1. **Reconnect to Colab** and mount Google Drive
2. **Pull latest code**: `!cd /content/nsga3-mammography-hpo-main && git pull`
3. **Load dataset** (Sections 5-7)
4. **Configure optimization** (Section 9-10)
5. **Run Section 11b** (Resume Optimization) instead of Section 11

The optimization will continue from the last saved checkpoint!

### Scenario 2: Need to Stop Early

You need to stop the optimization to free up resources:

1. **Click the Stop button** ⏹️ in Colab (or Runtime → Interrupt execution)
2. Wait for the current generation to finish (checkpoint will be saved)
3. Later, resume using Section 11b

### Scenario 3: Want to Extend Optimization

You ran 20 generations but want to run 10 more:

1. **Update config**: `config.nsga3.n_generations = 30`  (was 20, now 30)
2. **Run Section 11b** with `resume_from='auto'`
3. It will run 10 more generations (from 20 to 30)

### Scenario 4: Checkpoint Already Complete

If you try to resume but the checkpoint is already at or past your target:

```python
config.nsga3.n_generations = 20  # You want 20 generations
# But latest checkpoint is at generation 25

results = run_optimization(..., resume_from='auto')
```

**Output:**
```
Checkpoint already at or past target generations (25/20)
Loading final results from checkpoint...
```

The results from generation 25 will be returned immediately (no training).

## Checking Available Checkpoints

### In Colab

```python
import os
checkpoint_dir = "/content/drive/MyDrive/nsga3_outputs/checkpoints"

if os.path.exists(checkpoint_dir):
    checkpoints = sorted(os.listdir(checkpoint_dir))
    print(f"Available checkpoints ({len(checkpoints)}):")
    for cp in checkpoints:
        print(f"  - {cp}")
else:
    print("No checkpoints found")
```

### In Google Drive

Navigate to: `My Drive → nsga3_outputs → checkpoints/`

You'll see files like:
- `checkpoint_gen_5.pkl`
- `checkpoint_gen_10.pkl`
- `checkpoint_gen_15.pkl`
- etc.

## Tips & Best Practices

### 1. Keep Colab Active

Prevent Colab timeouts with this JavaScript snippet (paste in browser console):

```javascript
function ClickConnect(){
  console.log("Clicking connect");
  document.querySelector("colab-connect-button").click()
}
setInterval(ClickConnect, 60000)  // Click every 60 seconds
```

### 2. Monitor Progress

Checkpoints are saved with generation numbers. You can track progress:
- Generation 5 → ~5-10 minutes elapsed
- Generation 10 → ~10-20 minutes elapsed
- Generation 20 → ~20-40 minutes elapsed

(Times depend on population size and epochs)

### 3. Save Checkpoints More Frequently

For long runs, save checkpoints more often:

```python
config.nsga3.checkpoint_frequency = 2  # Every 2 generations instead of 5
```

**Trade-off**: More I/O overhead, but safer for interruptions

### 4. Clean Up Old Checkpoints

If disk space is limited, you can delete old checkpoints:

```python
import os
checkpoint_dir = "/content/drive/MyDrive/nsga3_outputs/checkpoints"

# Keep only the latest 3 checkpoints
checkpoints = sorted(os.listdir(checkpoint_dir))
for old_cp in checkpoints[:-3]:
    os.remove(os.path.join(checkpoint_dir, old_cp))
    print(f"Deleted: {old_cp}")
```

## Troubleshooting

### "No checkpoint found, starting fresh..."

**Cause**: No checkpoint files exist in the checkpoint directory

**Solutions**:
1. Check if you set the correct `OUTPUT_DIR`
2. Verify checkpoint directory exists: `/content/drive/MyDrive/nsga3_outputs/checkpoints/`
3. If this is your first run, this is normal (no resume needed)

### "Checkpoint not found: /path/to/file.pkl"

**Cause**: The specified checkpoint file doesn't exist

**Solutions**:
1. Check the file path is correct
2. List available checkpoints (see "Checking Available Checkpoints" above)
3. Use `resume_from='auto'` instead

### Resume seems to restart from scratch

**Cause**: Population might not be restored correctly

**Solutions**:
1. Check the console output - it should say "Restored population of size X"
2. Verify "Resuming NSGA-III: Completed X/Y generations"
3. If not showing, report an issue on GitHub

### Different results after resume

**Cause**: Random seed or configuration changed

**Solution**: Make sure you use the **same configuration** when resuming:
- Same `seed`
- Same `config.nsga3.pop_size`
- Same `config.hyperparameter_space`

## Advanced: Manual Checkpoint Loading

You can manually load and inspect checkpoints:

```python
import pickle

checkpoint_path = "/content/drive/MyDrive/nsga3_outputs/checkpoints/checkpoint_gen_10.pkl"

with open(checkpoint_path, 'rb') as f:
    checkpoint = pickle.load(f)

print(f"Generation: {checkpoint['generation']}")
print(f"Evaluations: {checkpoint['eval_count']}")
print(f"Population size: {len(checkpoint['population_X'])}")
print(f"History: {checkpoint['history']}")
```

## Summary

✅ **Checkpoints save automatically** every 5 generations
✅ **Resume with `resume_from='auto'`** to continue seamlessly
✅ **Checkpoints are saved to Google Drive** and persist across sessions
✅ **You can extend optimization** by increasing `n_generations` and resuming
✅ **No data loss** - full population and surrogate state are preserved

For questions or issues, please open an issue on GitHub!
