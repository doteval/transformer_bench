# Multi-GPU Transformer Operation Benchmarking Tool

A comprehensive benchmarking tool for evaluating transformer operations across different NVIDIA GPUs (RTX 3090, A10, H100). This tool measures the performance of key transformer operations including dense matrix multiplication and attention mechanisms across multiple precision formats.

## Features

- Support for multiple NVIDIA GPU architectures:
  - NVIDIA RTX 3090 (24GB GDDR6X)
  - NVIDIA A10 (24GB GDDR6)
  - NVIDIA H100 (80GB HBM3)
- Multiple precision formats:
  - FP16 (16-bit floating point)
  - FP32 (32-bit floating point)
  - BF16 (16-bit brain floating point)
  - INT8 (8-bit integer)
  - FP8 (8-bit floating point, H100 only)
- Benchmarks three key transformer operations:
  - Dense matrix multiplication
  - Query-Key attention initialization
  - Query-Key attention auto-regressive mode
- Customizable memory limits
- Comprehensive CSV and pickle output formats
- Progress tracking with detailed metrics

## Prerequisites

```bash
# Required Python packages
pip install torch numpy pandas tqdm
```

System requirements:
- Python 3.7+
- CUDA-compatible GPU
- PyTorch with CUDA support
- Sufficient disk space for output files

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/transformer-benchmarks.git
cd transformer-benchmarks
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

Run benchmarks on a specific GPU with default precision formats:
```bash
python benchmark.py --gpu a10     # For NVIDIA A10
python benchmark.py --gpu 3090    # For RTX 3090
python benchmark.py --gpu h100    # For H100
```

Run benchmarks with specific precision formats:
```bash
python benchmark.py --gpu h100 --dtype fp16 fp32 bf16   # Multiple precision formats
python benchmark.py --gpu 3090 --dtype fp16 int8        # Specific formats only
```

Run benchmarks on all supported GPUs:
```bash
python benchmark.py --all
python benchmark.py --all --dtype fp16 fp32  # All GPUs with specific formats
```

### Advanced Options

Override default memory limits:
```bash
python benchmark.py --gpu a10 --custom-memory 24  # Set custom memory limit in GB
```

### Command Line Arguments

| Argument | Description |
|----------|-------------|
| `--gpu` | Specify GPU type (choices: '3090', 'a10', 'h100') |
| `--dtype` | Specify precision formats to test (choices: 'fp16', 'fp32', 'bf16', 'int8', 'fp8') |
| `--all` | Run benchmarks on all available GPU types |
| `--custom-memory` | Override default GPU memory limit (in GB) |

## Output

The tool generates two types of output files in the `data/` directory:

1. CSV files (`transformer-batching-microbenchmarks-{gpu}-multi-dtype-{date}.csv`):
   - Detailed metrics for each operation
   - Performance statistics per precision format
   - Hardware-specific information

2. Pickle files (`{date}-transformer-batching-{gpu}-{dtype}.pkl.gz`):
   - Raw benchmark data per precision format
   - Complete measurement results
   - Compressed format for efficient storage

### Output Metrics

The benchmark results include:
- Latency measurements
- Throughput calculations
- FLOP counts
- Memory I/O statistics
- Arithmetic intensity
- Batch size scaling
- Sequence length impact
- Precision format performance comparisons

## Benchmark Configurations

Default configurations tested:
- Model dimensions: Various combinations of n∈[12,16,32,40,56,72,96] and d∈[64,128]
- Sequence lengths: [10, 20, 50, 100, 200, 500, 1000, 2000, 4000, 5000]
- Batch sizes: Range from 1 to 128 with variable increments
- Precision formats: FP16, FP32, BF16, INT8, FP8 (GPU-dependent)

## Data Analysis

The output CSV files can be analyzed using standard data analysis tools:

```python
import pandas as pd

# Load benchmark results
results = pd.read_csv('data/transformer-batching-microbenchmarks-a10-multi-dtype-20241124.csv')

# Basic analysis examples
throughput_stats = results.groupby(['series', 'dtype'])['throughput'].describe()
memory_efficiency = results.groupby(['series', 'bs', 'dtype'])['intensity'].mean()
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## License

MIT License (see LICENSE file for details)

## Citation

If you use this benchmark tool in your research, please cite:

```bibtex
@software{transformer_benchmarks_2024,
  title={Multi-GPU Transformer Operation Benchmarking Tool},
  author={Aakash Varma},
  year={2024},
  url={https://github.com/yourusername/transformer-benchmarks}
}
```

## Troubleshooting

Common issues and solutions:

1. **Out of Memory Errors**
   - Reduce batch sizes or sequence lengths
   - Use `--custom-memory` to set appropriate memory limits
   - Consider using lower precision formats

2. **CUDA Device Not Found**
   - Ensure CUDA toolkit is properly installed
   - Check GPU driver version compatibility
   - Verify PyTorch CUDA support with `torch.cuda.is_available()`

3. **Performance Issues**
   - Clear GPU cache between runs
   - Close other GPU-intensive applications
   - Monitor GPU temperature and throttling

4. **Precision Format Compatibility**
   - Verify GPU supports requested precision formats
   - Check PyTorch version supports desired precision
   - H100-specific features (like FP8) require appropriate hardware

## Support

For bug reports and feature requests, please use the GitHub issue tracker.

For questions and discussions, feel free to reach out to @varmology on X (previously Twitter).
