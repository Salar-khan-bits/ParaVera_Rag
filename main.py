import requests

response = requests.post(
    "http://localhost:8080/v1/chat/completions",
    json={
        "messages": [
            {"role": "user", "content": "Hi there"}
        ],
        "temperature": 0.7,
        "max_tokens": 300
    }
)

print(response.json()["choices"][0]["message"]["content"])


