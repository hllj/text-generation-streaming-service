# Offline Benchmark

## Float16

Device: A100 PCIE

| Backend  | Token/s | Requests/s |
|----------|---------|------------|
| vLLM - float16     | 2080.24 | 10.49      |
| HF - float16       | 294.42  | 1.48       |
| TensorRT - float16 | 164.16  | 0.83       |
| TensorRT - int8    | 249.56  | 1.26       |

Script:

```bash
python3 src/benchmark_throughput.py \
--backend vllm \
--model vinai/PhoGPT-7B5-Instruct \
--tokenizer vinai/PhoGPT-7B5-Instruct \
--n 1 \
--num-requests 100 \
--seed 42 \
--dtype float16 \
--max_output_len 1024
```

```bash
python3 src/benchmark_throughput.py \
--backend hf \
--model vinai/PhoGPT-7B5-Instruct \
--tokenizer vinai/PhoGPT-7B5-Instruct \
--n 1 \
--num-requests 100 \
--seed 42 \
--dtype float16 \
--max_output_len 1024 \
--max-batch-size 8
```

```bash
python3 src/benchmark_throughput.py \
--backend tensorrt \
--engine_dir trt_engines/phogpt-7b5-instruct/fp16/1-gpu/ \
--use_py_session \
--max_input_len 1024 \
--max_output_len 1024 \
--tokenizer_dir vinai/PhoGPT-7B5-Instruct \
--max-batch-size 16 \
--num-requests 100
```

# Online Benchmark - Serving

Test with 1000 prompts from dataset.

Device: A100 PCIE

| Backend + Config   | Throughput (req/s) | Avg latecy (s) | Avg latency per token (s) | Avg latency per output token (s) |
|--------------------|--------------------|----------------|---------------------------|----------------------------------|
| vLLM - float32     | 2.15               | 230.08         | 1.62                      | 6.76                             |
| vLLM - float16     | 15.15              | 32.89          | 0.20                      | 0.71                             |
| TensorRT - float16 | 6.85               | 69.78          | 0.50                      | 2.15                             |
| TensorRT - int8    | 8.90               | 57.36          | 0.42                      | 1.81                             |

```bash
python3 src/benchmark_serving.py \
--backend vllm \
--seed 42 \
--tokenizer vinai/PhoGPT-7B5-Instruct \
--num-requests 1000
```

```bash
python3 src/benchmark_serving.py \
--backend tensorrt \
--seed 42 \
--tokenizer vinai/PhoGPT-7B5-Instruct \
--num-requests 1000
```