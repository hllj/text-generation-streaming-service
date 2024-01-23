# Text Generation Streaming Service

## Installation

```bash
pip install -r requirements.txt
```

## Start service

```bash
bash start_server.sh
```

or

```bash
python server.py \
--host 127.0.0.1 \
--port 3000 \
--model vinai/PhoGPT-7B5-Instruct \
--tokenizer vinai/PhoGPT-7B5-Instruct \
--load-format auto \
--dtype bfloat16 \
--seed 42
```

## Test