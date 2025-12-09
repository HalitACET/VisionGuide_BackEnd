import requests
import base64
import json
import os
import time

def test_ocr():
    # 1. Image file path
    image_path = "metin.jpg"
    
    if not os.path.exists(image_path):
        print(f"Error: {image_path} not found.")
        return

    # 2. Read and encode image
    with open(image_path, "rb") as f:
        image_data = f.read()
        base64_image = base64.b64encode(image_data).decode('utf-8')

    # 3. Prepare payload
    payload = {
        "image": base64_image
    }

    # 4. Send request
    url = "http://localhost:8000/ocr"
    print(f"Sending OCR request to {url}...")
    
    max_retries = 5
    for i in range(max_retries):
        try:
            response = requests.post(url, json=payload)
            
            if response.status_code == 200:
                print("\nSuccess! OCR Result:")
                print(json.dumps(response.json(), indent=2, ensure_ascii=False))
                return
            else:
                print(f"\nError: Status Code {response.status_code}")
                print(response.text)
                time.sleep(1)
        except requests.exceptions.ConnectionError:
            print(f"Connection failed, retrying... ({i+1}/{max_retries})")
            time.sleep(2)

if __name__ == "__main__":
    test_ocr()
