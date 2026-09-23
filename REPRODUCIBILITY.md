# Reproducibility details

This file consolidates the exact model configurations, software versions,
hardware, and determinism settings behind the reported measurements. Every
value is taken from the committed configuration files, the pinned
`requirements.txt`, or the `environment.json` capture that each run writes.

## Model architectures

The from-scratch grid trains four classifier formulations at three jointly
scaled sizes. The exact dimensions are defined in `pls_train_classifier.py`
and reproduced here.

| Size | Hidden size | Layers | Attention heads | Feed-forward width |
|---|---|---|---|---|
| small | 64 | 1 | 1 | 256 |
| med | 384 | 6 | 6 | 1,536 |
| large | 768 | 12 | 12 | 3,072 |

The BERT-style encoder uses `hidden_size`, `num_hidden_layers`,
`num_attention_heads`, and `intermediate_size`; the GPT-2-style decoder uses
`n_embd`, `n_layer`, and `n_head` with the default 4x feed-forward
multiplier. The `pretrained` tier loads the public 768-dimensional
checkpoints unchanged. Maximum sequence length, dataset list, seeds, and the
full 180-job plan are in `configs/study_180.yaml` and can be verified with
`python pls_plan_train.py --config configs/study_180.yaml --explain-counts`.

## Software versions

Pinned installation versions are in `requirements.txt`:
`torch==2.11.0+cu130`, `transformers==5.6.2`, `datasets==4.8.4`,
`numpy==2.4.4`, `scikit-learn==1.8.0`, `scipy==1.17.1`, `pandas==3.0.2`,
`PyYAML==6.0.3`, `pyarrow==24.0.0`.

The runtime capture recorded with each artifact reports Python 3.12.13,
PyTorch 2.11.0+cu130 built against CUDA 13.0 with cuDNN 9.19.0, Intel MKL
2024.2 with MKL-DNN 3.10.2 for CPU BLAS, and Transformers 5.6.2. Each run
writes its own `environment.json` with these fields, so any published number
can be checked against the library versions that produced it.

## Hardware

GPU measurements use one NVIDIA L40S (compute capability 8.9). CPU
measurements use an AMD EPYC 7R13 host; the thread count is controlled
explicitly (`torch.set_num_threads`), and runs record the value used.

## Determinism settings

Unless a run declares otherwise in its manifest, inference sets
`CUBLAS_WORKSPACE_CONFIG=:4096:8`, `torch.backends.cudnn.deterministic=True`,
`torch.backends.cudnn.benchmark=False`, TF32 disabled for matrix multiply and
cuDNN, `PYTHONHASHSEED=0`, Python, NumPy, and Torch seeds 0, single-threaded
Torch intra-op and inter-op execution, and `TRANSFORMERS_OFFLINE=1`. The
same-context repeatability floor reported in the paper is measured under
these settings.

## Quantization

Dynamic 8-bit integer quantization on CPU uses
`torch.quantization.quantize_dynamic` over `torch.nn.Linear` modules with
`dtype=torch.qint8`, which computes the activation scale per batch at run
time. Static 8-bit quantization fixes a per-tensor activation scale
calibrated offline before serving, with per-channel int8 weights over the
same module set, so the scale does not depend on the batch content. The
GPU-held int8 comparison uses the torchao int8 dynamic quantization API on
the same checkpoints. For the GPT-2 family, only the language-model head is
an `nn.Linear` module (the Hugging Face implementation uses `Conv1D`
elsewhere), so dynamic quantization covers that projection.
