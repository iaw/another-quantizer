# GLM Sequential Quantization

Simple AWQ 4-bit quantization for GLM models with memory-efficient sequential processing.

## Features

- Sequential layer-by-layer quantization (low memory usage)
- Resumable checkpointing
- INT4 weight quantization
- Support for large models that don't fit in GPU memory

## Install

```bash
pip install torch transformers safetensors tqdm psutil pyyaml
```

## Quick Start

### Basic Usage

```bash
# Quantize a model
python main.py --model-path /path/to/glm-model --output-path /path/to/output --bits 4

# Resume from checkpoint
python main.py --resume --checkpoint-dir /path/to/checkpoints

# Custom settings
python main.py --model-path /path/to/glm-model \
               --output-path /path/to/output \
               --bits 4 \
               --group-size 128 \
               --calibration-samples 128 \
               --checkpoint-every-n-layers 5
```

### Using Config File

```bash
# Create config file (see config_example.yaml)
python main.py --config config.yaml
```

## Options

| Option | Default | Description |
|--------|---------|-------------|
| `--model-path` | required | Path to GLM model |
| `--output-path` | `{model_path}_quantized` | Output directory |
| `--bits` | 4 | Quantization bits (3, 4, or 8) |
| `--group-size` | 128 | Group size for quantization |
| `--calibration-samples` | 128 | Number of calibration samples |
| `--batch-size` | 2 | Calibration batch size |
| `--checkpoint-dir` | `{output_path}/checkpoints` | Checkpoint directory |
| `--checkpoint-frequency` | 5 | Layers between checkpoints |
| `--max-gpu-memory` | auto-detect | GPU memory limit (GB) |
| `--max-cpu-memory` | auto-detect | CPU memory limit (GB) |
| `--offload-to-cpu` | true | Enable CPU offloading |
| `--resume` | false | Resume from latest checkpoint |

## Memory Requirements

| Model Size | GPU Memory | CPU Memory |
|------------|------------|------------|
| 7B | 8-16 GB | 32 GB |
| 13B | 16-24 GB | 64 GB |
| 30B+ | 24-40 GB | 128 GB |

## Supported Models

- GLM-4
- ChatGLM3
- GLM-4.5 (including MoE variants)

## Troubleshooting

### Out of Memory

- Reduce `--batch-size` to 1
- Enable `--offload-to-cpu`
- Increase checkpoint frequency

### Slow Quantization

- Increase `--batch-size` if memory allows
- Use GPU if available
- Reduce `--calibration-samples` for faster (but potentially lower quality) quantization

### Resume Failed

Check that checkpoint directory contains `checkpoint_state.json` file.

## Output Format

The quantized model will be saved in safetensors format with:
- Quantized weights (INT4 packed)
- Scale and zero-point parameters
- Original model configuration
- Tokenizer files

## License

Apache 2.0