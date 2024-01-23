import requests
import json

url = "http://localhost:8000/generate"

payload = json.dumps({
  "prompt": "### Câu hỏi:\nViết bài văn nghị luận xã hội về an toàn giao thông\n\n### Trả lời:",
  "stream": True,
  "n": 1,
  "best_of": 1,
  "temperature": 1,
  "top_k": 50,
  "top_p": 0.9,
  "max_tokens": 1024,
  "stop_token_ids": [
    2
  ]
})
headers = {
  'Content-Type': 'application/json',
  'Cookie': 'redirect_to=%2Fgenerate'
}

for i in range(100):
    response = requests.request("POST", url, headers=headers, data=payload)

    print(response.text)