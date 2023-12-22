# gpt-fast Testing

This page documents various experiments with [gpt-fast](https://github.com/pytorch-labs/gpt-fast), a pure PyTorch benchmark for optimizing auto-regressive models like Llama.

## Results

**Baseline** : Asclepius-13B, AWQ, 1 GPU, vLLM | 60.30 Avg Token / Sec

**Model**: Asclepius-13B
**Environment**: CUDA 12.1, 4 A100 40GB GPUs

| Quantization | model.compile() | Tensor Parallel / GPUs | Avg Token / Sec |
| --- | --- | --- | --- |
| int8 | No | No / 1 GPU | 10.96 |
| int8 | Yes | No / 1 GPU | 42.22 |
| int8 | Yes | Yes / 2 GPU | 76.79 | 
| int8 | Yes | Yes / 4 GPU | 114.01 |
| int4 | No | No / 1 GPU | 28.43 |
| int4 | Yes | No / 1 GPU | 64.47 |
| int4 | Yes | Yes / 2 GPU | 111.60 |
| int4 | Yes | Yes / 4 GPU | **141.45** |

**Model**: upstage/Llama-2-70b-instruct
**Environment**: CUDA 12.1, 4 A100 40GB GPUs

| Quantization | model.compile() | Tensor Parallel / GPUs | Speculative Decoding | Avg Token / Sec |
| int8 | - | - | - | Did not fit on single GPU |
| int4 | No | No / 1 GPU | No | 10.64 |
| int4 | Yes | No / 1 GPU | No | 12.71 |
| int4 | Yes | Yes / 2 GPU | No | 23.86 |
| int4 | Yes | Yes / 4 GPU | No | **38.91** |
| int4 | Yes | No / 1 GPU | Yes Asclepius-13B 4-bit | Did not fit on single GPU |
| int4 | Yes | Yes / 2 GPU | Yes Asclepius-13B 4-bit | 29.79 (lots of variance) |
| int4 | Yes | Yes / 4 GPU | Yes Asclepius-13B 4-bit | 30.94 |
| int4 | Yes | Yes / 4 GPU | Yes Asclepius-13B 8-bit | 33.76 |

## Conclusions

The only performance metric used here is average tokens per second. It is assumed that there is no significant drop in validation accuracy for any of these mechanisms. That is a naive assumption and should be tested in the future.

### model.compile

For inference, model.compile should be default. This speed up is consistent and obvious.

### Quantization

When working with models in the 70B param range, using int4 is necessary to fit into a single GPU. With smaller models, it may be sufficient to use int8 to avoid any unnecessary loss in validation accuracy.

### Tensor Parallel

There is near ideal scaling when adding GPUs with tensor parallelism with likely little to no drop in validation accuracy. If multiple GPUs are available and latency is critical, this is a good speedup to use.

### Speculative Decoding

This approach did not improve the average tokens per second; in fact, there was enough overhead to lose some performance.

There is an interesting trend here: a higher precision (better) small/cheap model leads to a better tokens per second value. I believe this is because the bigger/expensive model needs to step in fewer times and the performance is better.

### Combinations

The best combination of optimizations for both models is model.compile, int4-quantization, and tensor parallelism over the maximum number of GPUs. Speculative decoding did not improve the average tokens per second metric.
