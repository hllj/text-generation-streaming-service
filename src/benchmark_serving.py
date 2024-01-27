"""Benchmark online serving throughput."""
import random
import numpy as np
import argparse
import asyncio
import aiohttp
import requests
import json
import time
from tqdm.asyncio import tqdm
from typing import AsyncGenerator, List, Tuple

from transformers import (AutoConfig, AutoModelForCausalLM, AutoTokenizer,
                          PreTrainedTokenizerBase)

from utils import sample_requests

# (prompt len, output len, latency)
REQUEST_LATENCY: List[Tuple[int, int, float]] = []

async def send_request(api_url, prompt, prompt_len, output_len, backend, best_of, temperature, top_k, top_p, max_new_tokens, pbar):
    request_start_time = time.perf_counter()
    payload = {
        "prompt": prompt,
        "stream": True,
        "n": 1,
        "best_of": 1,
        "temperature": 1,
        "top_k": 50,
        "top_p": 0.9
    }
    if backend == 'vllm':
        payload["max_tokens"] = output_len
        payload["stop_token_ids"] = [2]
    elif backend == 'tensorrt':
        payload["max_num_tokens"] = output_len
    elif backend == 'hf':
        payload["max_new_tokens"] = output_len
    headers = {
        "User-Agent": "Benchmark Client"
    }
    timeout = aiohttp.ClientTimeout(total=3 * 3600)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(api_url, headers=headers,
                                json=payload) as response:
            chunks = []
            async for chunk, _ in response.content.iter_chunks():
                chunks.append(chunk)
        output = chunks[-1].decode("utf-8")
        print(output)
        
    request_end_time = time.perf_counter()
    request_latency = request_end_time - request_start_time

    REQUEST_LATENCY.append((prompt_len, output_len, request_latency))

    pbar.update(1)

async def get_request(requests, request_rate) -> AsyncGenerator[Tuple[str, int, int], None]:
    requests = iter(requests)
    for request in requests:
        yield request

        if request_rate == float("inf"):
            # If the request rate is infinity, then we don't need to wait.
            continue
        # Sample the request interval from the exponential distribution.
        interval = np.random.exponential(1.0 / request_rate)
        # The next request will be sent after the interval.
        await asyncio.sleep(interval)

async def benchmark(
    api_url,
    requests,
    backend,
    n,
    best_of, 
    temperature, 
    top_k, 
    top_p, 
    max_new_tokens,
    request_rate
):
    tasks = []
    pbar = tqdm(total=len(requests))
    async for request in get_request(requests, request_rate):
        prompt, prompt_len, completion, completion_len = request
        task = asyncio.create_task(
            send_request(api_url, prompt, prompt_len, completion_len, backend, best_of, temperature, top_k, top_p, max_new_tokens, pbar)
        )
        tasks.append(task)
    await asyncio.gather(*tasks)
    pbar.close()

def main(args):
    print(args)
    random.seed(args.seed)
    np.random.seed(args.seed)

    api_url = f"{args.protocol}://{args.host}:{args.port}{args.endpoint}"
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, trust_remote_code=args.trust_remote_code)
    
    requests = sample_requests('benchmark/vi.json', args.num_requests, tokenizer)

    start = time.perf_counter()

    asyncio.run(
        benchmark(
            api_url, requests, 
            args.backend, args.n, args.best_of, args.temperature, 
            args.top_k, args.top_p, args.max_new_tokens, 
            args.request_rate
        )
    )

    end = time.perf_counter()
    benchmark_time = end - start
    print(f"Total time: {benchmark_time:.2f} s")
    print(f"Throughput: {args.num_requests / benchmark_time:.2f} requests/s")

    # Compute the latency statistics.
    avg_latency = np.mean([latency for _, _, latency in REQUEST_LATENCY])
    print(f"Average latency: {avg_latency:.2f} s")
    avg_per_token_latency = np.mean([
        latency / (prompt_len + output_len)
        for prompt_len, output_len, latency in REQUEST_LATENCY
    ])
    print(f"Average latency per token: {avg_per_token_latency:.2f} s")
    avg_per_output_token_latency = np.mean(
        [latency / output_len for _, output_len, latency in REQUEST_LATENCY])
    print("Average latency per output token: "
          f"{avg_per_output_token_latency:.2f} s")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Benchmark the online serving throughput.")
    parser.add_argument("--backend",
                        type=str,
                        default="vllm",
                        choices=["vllm", "tensorrt", "hf"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tokenizer", type=str, default="vinai/PhoGPT-7B5-Instruct")
    parser.add_argument("--protocol",
                        type=str,
                        default="http",
                        choices=["http", "https"])
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--endpoint", type=str, default="/generate")

    parser.add_argument("--n", type=int, default=1)
    parser.add_argument("--best-of", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-new-tokens", type=int, default=1024)

    parser.add_argument('--trust-remote-code',
                        action='store_true',
                        help='trust remote code from huggingface')
    parser.add_argument("--request-rate",
                        type=float,
                        default=float("inf"),
                        help="Number of requests per second. If this is inf, "
                        "then all the requests are sent at time 0. "
                        "Otherwise, we use Poisson process to synthesize "
                        "the request arrival times.")
    parser.add_argument("--num-requests",
                        type=int,
                        default=100,
                        help="Number of prompts to process.")

    args = parser.parse_args()
    main(args)