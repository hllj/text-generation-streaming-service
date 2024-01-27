from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread

class AsyncModel():
    def __init__(self, model_path, trust_remote_code, dtype, device):
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=trust_remote_code)
        self.device = device
        config.init_device = device
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, config=config, 
            torch_dtype=dtype, trust_remote_code=trust_remote_code
        )
        self.model.to(device)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=trust_remote_code)

    def async_generate(self, prompt, **kwargs):
        input_ids = self.tokenizer(prompt, return_tensors="pt")
        streamer = TextIteratorStreamer(self.tokenizer)
        generation_kwargs = dict(
            inputs=input_ids["input_ids"].to(self.device),
            attention_mask=input_ids["attention_mask"].to(self.device),
            streamer=streamer,
            **kwargs
        )
        # self.model.generate(**generation_kwargs)
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()
        for text in streamer:
            yield text

    def generate(self, prompt, **kwargs):
        input_ids = self.tokenizer(prompt, return_tensors="pt")
        generation_kwargs = dict(
            inputs=input_ids["input_ids"].to(self.device),
            attention_mask=input_ids["attention_mask"].to(self.device),
            **kwargs
        )

        output = self.model.generate(**generation_kwargs)
        output = self.tokenizer.batch_decode(output, skip_special_tokens=True)[0]
        return output
