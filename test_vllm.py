from vllm import LLM, SamplingParams

seed = 42
model_path = "vinai/PhoGPT-7B5-Instruct"

sampling_params = SamplingParams(
    n=1,
    best_of=1,
    temperature=1.0,
    top_k=50,
    top_p=0.9,
    max_tokens=1024,
    stop_token_ids=[2]
)
llm = LLM(model=model_path, seed=seed)


PROMPT_TEMPLATE = "### Câu hỏi:\n{instruction}\n\n### Trả lời:"  

# Some instruction examples
# instruction = "Viết bài văn nghị luận xã hội về {topic}"
# instruction = "Viết bản mô tả công việc cho vị trí {job_title}"
# instruction = "Sửa lỗi chính tả:\n{sentence_or_paragraph}"
# instruction = "Dựa vào văn bản sau đây:\n{text}\nHãy trả lời câu hỏi: {question}"
# instruction = "Tóm tắt văn bản:\n{text}"


instruction = "Viết bài văn nghị luận xã hội về an toàn giao thông"
# instruction = "Sửa lỗi chính tả:\nTriệt phá băng nhóm kướp ô tô, sử dụng \"vũ khí nóng\""

input_prompt = PROMPT_TEMPLATE.format_map(  
    {"instruction": instruction}  
)

output = llm.generate(input_prompt, sampling_params)[0]

text = output.outputs[0].text

print(text)