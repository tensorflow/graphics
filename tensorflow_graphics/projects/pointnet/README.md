# PointNet v1.0 Implementation

## Dependencies

```
pip install h5py
pip install pytest
pip install termcolor
pip install tensorflow==2.1.0
pip install tensorflow_graphics
pip install tensorflow_datasets
```

## Training time
- Approximately 51m for 250 epochs (vs. 2h50m of the legacy TF1 implementation)
- CPU: Intel(R) Xeon(R) CPU @ 2.30GHz, 4 cores
- GPU: NVIDIA Tesla V100, Driver Version: 440.64.00, CUDA Version: 10.2

## Classification benchmarks
- TF1 (legacy PointNet), Batch 2048 → 86.77 @ 73k steps
- TF2 (this repository), Batch 2048 → 87.60 @ 36k steps

## MISC
The code is this folder is not compatible with Graph mode.
