# Text Generation Streaming Service

## Installation

```bash
pip install -r requirements.txt
```

## Start vLLM service

```bash
bash start_server.sh
```

or

```bash
python services/vllm/server.py \
--host 127.0.0.1 \
--port 8000 \
--model vinai/PhoGPT-7B5-Instruct \
--tokenizer vinai/PhoGPT-7B5-Instruct \
--load-format auto \
--dtype bfloat16 \
--seed 42
```

### Quantization

## Start TensorRT service

## Convert HF weight

Follow instruction in /services/tensorrt README.

### Quantization

## Test