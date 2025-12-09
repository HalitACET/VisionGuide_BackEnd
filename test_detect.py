import base64
import json
import requests

# 1) Test resmi oku
with open("test.jpeg", "rb") as f:  # backend klasöründe test.jpg olsun
    b64 = base64.b64encode(f.read()).decode("utf-8")

# 2) İsteği hazırla
url = "http://localhost:8000/detect"
payload = {"image": b64}

# 3) POST isteği gönder
r = requests.post(url, json=payload)
print("Status:", r.status_code)
print("Body:", json.dumps(r.json(), indent=2, ensure_ascii=False))