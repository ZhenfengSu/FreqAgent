import requests
import json

base_url = "http://localhost:6001"

# 先获取可用的模型名称
models_response = requests.get(f"{base_url}/v1/models")
print("=== Available Models ===")
print(json.dumps(models_response.json(), indent=2))

# 使用模型名称请求
model_name = models_response.json()['data'][0]['id']  # 获取第一个模型名

response = requests.post(
    f"{base_url}/v1/completions",
    json={
        "model": model_name,
        "prompt": "Hello, how are you?",
        "max_tokens": 10,
        "temperature": 0,
        "logprobs": 5,
    },
    timeout=30
)

result = response.json()
print("\n=== Completion Response ===")
print(json.dumps(result, indent=2, default=str))