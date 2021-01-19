import requests

url = "http://localhost:5002/api/detect"

payload = {"token": "12345678", "alg": "gmm", "audio_path": "Cough1.wav"}

response = requests.post(url, json=payload)

print(response.json())
