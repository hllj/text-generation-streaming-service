import argparse
import json
from typing import AsyncGenerator

import torch

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
import uvicorn

from model import AsyncModel

from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator

TIMEOUT_KEEP_ALIVE = 5  # seconds.
app = FastAPI()
instrumentator = Instrumentator().instrument(app)
engine = None

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health() -> Response:
    """Health check."""
    return Response(status_code=200)

@app.on_event("startup")
async def _startup():
    instrumentator.expose(app)


@app.post("/generate")
async def generate(request: Request) -> Response:
    """Generate completion for the request.

    The request should be a JSON object with the following fields:
    - prompt: the prompt to use for the generation.
    - stream: whether to stream the results or not.
    - other fields: the sampling parameters (See `SamplingParams` for details).
    """
    request_dict = await request.json()
    prompt = request_dict.pop("prompt")
    # prefix_pos = request_dict.pop("prefix_pos", None)
    stream = request_dict.pop("stream", False)

    kwargs = dict(
        do_sample=True,
        temperature=request_dict.pop("temperature", 1.0),  
        top_k=request_dict.pop("top_k", 50),  
        top_p=request_dict.pop("top_p", 0.9),  
        max_new_tokens=request_dict.pop("max_new_tokens", 100),
        eos_token_id=engine.tokenizer.eos_token_id,  
        pad_token_id=engine.tokenizer.pad_token_id,
        use_cache=True
    )

    generator = engine.async_generate(prompt, **kwargs)

    # Streaming case
    async def stream_results():
        output_text = ""
        async for text in generator:
            output_text += text
            ret = {"text": prompt + output_text}
            yield (json.dumps(ret) + "\0").encode("utf-8")

    if stream:
        return StreamingResponse(stream_results())

    # Non-streaming case
    output_text = engine.generate(prompt, **kwargs)
    ret = {"text": prompt + output_text}
    return JSONResponse(ret)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default=None)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--ssl-keyfile", type=str, default=None)
    parser.add_argument("--ssl-certfile", type=str, default=None)
    parser.add_argument(
        "--root-path",
        type=str,
        default=None,
        help="FastAPI root_path when app is behind a path based routing proxy")
    
    parser.add_argument("--model-path", type=str, default="vinai/PhoGPT-7B5-Instruct")
    parser.add_argument("--trust-remote-code", default=True)
    parser.add_argument("--device", type=str, default="cuda", choices=['float16', 'float32', 'bfloat16'])
    parser.add_argument("--dtype", type=str, default="float16")
    args = parser.parse_args()

    if args.dtype == 'float16':
        args.dtype = torch.float16
    elif args.dtype == 'float32':
        args.dtype = torch.float32
    elif args.dtype == 'bfloat16':
        args.dtype = torch.bfloat16

    engine = AsyncModel(args.model_path, args.trust_remote_code, args.dtype, args.device)

    app.root_path = args.root_path
    uvicorn.run(app,
                host=args.host,
                port=args.port,
                log_level="debug",
                timeout_keep_alive=TIMEOUT_KEEP_ALIVE,
                ssl_keyfile=args.ssl_keyfile,
                ssl_certfile=args.ssl_certfile)