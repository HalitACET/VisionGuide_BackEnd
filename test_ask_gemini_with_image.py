import requests
import base64
import json
import os

import time

def test_gemini_endpoint():
    # 1. Image file path
    image_path = "smartphone.png"
    
    if not os.path.exists(image_path):
        print(f"Error: {image_path} not found.")
        return

    # 2. Read and encode image
    with open(image_path, "rb") as f:
        image_data = f.read()
        base64_image = base64.b64encode(image_data).decode('utf-8')

    # 3. Prepare payload
    payload = {
        "image": base64_image,
        "prompt": "Bu resimde ne görüyorsun? Kalem nerede?"
    }

    # 4. Send request with retry
    url = "http://localhost:8000/ask_gemini"
    print(f"Sending request to {url}...")
    
    max_retries = 10
    for i in range(max_retries):
        try:
            response = requests.post(url, json=payload)
            
            if response.status_code == 200:
                print("\nSuccess! Response:")
                print(json.dumps(response.json(), indent=2, ensure_ascii=False))
                return
            else:
                print(f"\nError: Status Code {response.status_code}")
                print(response.text)
                return
                
        except requests.exceptions.ConnectionError:
            print(f"Connection failed, retrying in 2 seconds... ({i+1}/{max_retries})")
            time.sleep(2)
    
    print("\nError: Could not connect to the server after multiple attempts.")

if __name__ == "__main__":
    test_gemini_endpoint()
