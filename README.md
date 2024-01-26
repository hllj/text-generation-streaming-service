# Text Generation Streaming Service

## Installation and Setup

### Installation libraries

1. Docker base image: : nvidia/cuda:12.3.1-devel-ubuntu22.04

2. Install TensorRT-LLM

```bash
# Install dependencies, TensorRT-LLM requires Python 3.10
apt-get update && apt-get -y install python3.10 python3-pip openmpi-bin libopenmpi-dev

# Install the latest preview version (corresponding to the main branch) of TensorRT-LLM.
# If you want to install the stable version (corresponding to the release branch), please
# remove the `--pre` option.
pip3 install tensorrt_llm -U --pre --extra-index-url https://pypi.nvidia.com

# Check installation
python3 -c "import tensorrt_llm; print(tensorrt_llm.__version__)"
```

3. Install vLLM and other libraries

```bash
pip3 install -r requirements.txt
```

4. Login HuggingFace with Read Token

```bash
huggingface-cli login
```

## Start vLLM service

```bash
bash start_vllm_server.sh
```

or

```bash
python3 src/services/vllm/server.py \
--host 127.0.0.1 \
--port 8000 \
--model vinai/PhoGPT-7B5-Instruct \
--tokenizer vinai/PhoGPT-7B5-Instruct \
--load-format auto \
--dtype bfloat16 \
--seed 42
```

## TensorRT service

### Quantization

## Convert HF weight to Float16

Follow instruction in /services/tensorrt README.

### Convert to FT format in dtype float32/float16/bfloat16
```bash
python3 src/services/tensorrt/convert_hf_mpt_to_ft.py \
-i vinai/PhoGPT-7B5-Instruct \
-o ./weights/ft/phogpt-7b5-instruct/fp16/ \
-t float16
```

### Build Engine

```bash
python3 src/services/tensorrt/build.py \
--model_dir=./weights/ft/phogpt-7b5-instruct/fp16/1-gpu/ \
--max_batch_size 16 \
--use_inflight_batching \
--paged_kv_cache \
--use_gpt_attention_plugin \
--use_gemm_plugin \
--output_dir ./trt_engines/phogpt-7b5-instruct/fp16/1-gpu/ \
--max_input_len 1024 \
--max_output_len 1024 \
--max_num_tokens 2048
```

### Run a test

```bash
python3 src/services/tensorrt/test.py \
--max_output_len 1024 \
--engine_dir ./trt_engines/phogpt-7b5-instruct/fp16/1-gpu/ \
--tokenizer_dir vinai/PhoGPT-7B5-Instruct \
--input_text "### Câu hỏi:\nViết bài văn nghị luận xã hội về an toàn giao thông\n\n### Trả lời:" \
--streaming \
--use_py_session
```

### Start TensorRT service

```bash
bash start_tensorrt_server.sh
```

or 

```bash
python3 src/services/tensorrt/server.py trt_engines/phogpt-7b5-instruct/fp16/1-gpu/ vinai/PhoGPT-7B5-Instruct \
--host 127.0.0.1 \
--port 8000
```