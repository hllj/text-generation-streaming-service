from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig
import torch
from benchmark import sample_requests

model_id = "vinai/PhoGPT-7B5-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)

requests = sample_requests('benchmark/vi.json', 128, tokenizer)

datasets = [prompt + '\n' + completion for (prompt, prompt_len, completion, output_len) in requests]

quantization_config = GPTQConfig(
     bits=4,
     group_size=128,
     dataset=datasets,
     desc_act=False,
     block_name_to_quantize = "model.decoder.layers"
)


quant_model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=quantization_config, device_map='auto')

text = "### Câu hỏi:\nViết bài văn nghị luận xã hội về an toàn giao thông\n\n### Trả lời:"
inputs = tokenizer(text, return_tensors="pt").to(0)

out = quant_model.generate(**inputs)
print(tokenizer.decode(out[0], skip_special_tokens=True))