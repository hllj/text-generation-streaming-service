python3 src/services/vllm/server.py \
--host 127.0.0.1 \
--port 8000 \
--model vinai/PhoGPT-7B5-Instruct \
--tokenizer vinai/PhoGPT-7B5-Instruct \
--load-format auto \
--dtype float16 \
--seed 42