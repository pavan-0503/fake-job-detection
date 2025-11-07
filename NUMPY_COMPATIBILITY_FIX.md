# NumPy Compatibility Fix

## Problem
The application was crashing on Railway with two related errors:

### Error 1: NumPy Version Incompatibility
```
A module that was compiled using NumPy 1.x cannot be run in
NumPy 2.3.4 as it may crash. To support both 1.x and 2.x
versions of NumPy, modules must be compiled with NumPy 2.0.
```

### Error 2: PyTorch/Transformers Incompatibility
```
AttributeError: module 'torch.utils._pytree' has no attribute 'register_pytree_node'
```

## Root Cause
1. **NumPy 2.x Breaking Changes**: PyTorch 2.1.0 and scikit-learn were compiled against NumPy 1.x, but pip was installing NumPy 2.3.4 (latest version)
2. **Transformers Version**: The latest transformers library expects newer PyTorch APIs that don't exist in PyTorch 2.1.0+cpu

## Solution Applied
Changed `requirements.txt` to pin compatible versions:

```diff
- transformers>=4.30.0
+ transformers==4.35.2

- numpy>=1.24.0
+ numpy<2.0.0
```

### Why These Versions?
- **numpy<2.0.0**: Ensures NumPy 1.x (1.24-1.26) is installed, which is compatible with PyTorch 2.1.0+cpu
- **transformers==4.35.2**: Last known stable version that works well with PyTorch 2.1.0 and NumPy 1.x

## Expected Outcome
✅ Build will complete successfully in ~5-7 minutes  
✅ NumPy 1.26.x will be installed (compatible with PyTorch)  
✅ No more "compiled using NumPy 1.x" warnings  
✅ No more `register_pytree_node` AttributeError  
✅ Gunicorn workers will boot successfully  
✅ App will run without crashes

## Timeline
- **Previous fix (pandas)**: Commit 3a65bca  
- **This fix (numpy/transformers)**: Commit 9849380  
- **Railway deployment**: Auto-triggered, ~7-10 minutes total

## Monitoring
Check Railway logs for:
```
Successfully installed numpy-1.26.x transformers-4.35.2
[INFO] Starting gunicorn 23.0.0
[INFO] Booting worker with pid: X
```

Should NOT see:
- ❌ "A module that was compiled using NumPy 1.x..."
- ❌ "AttributeError: module 'torch.utils._pytree'..."
- ❌ "Worker failed to boot"

## Lesson Learned
When using CPU-only PyTorch, must ensure:
1. NumPy version matches what PyTorch was compiled against (check PyTorch docs)
2. Transformers version is compatible with specific PyTorch version
3. Use exact version pins (`==`) for critical ML libraries, not ranges (`>=`)
