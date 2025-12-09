import requests
import base64
import json
import os
import time

def test_currency():
    # 1. Image file path
    image_path = "para.jpg"
    
    if not os.path.exists(image_path):
        # Fallback to desk image if specific currency image generation fails or is not moved
        print(f"Warning: {image_path} not found. Trying test_desk_with_pen.png as generic test.")
        image_path = "test_desk_with_pen.png"
        if not os.path.exists(image_path):
             print("Error: No test image found.")
             return

    # 2. Read and encode image
    with open(image_path, "rb") as f:
        image_data = f.read()
        base64_image = base64.b64encode(image_data).decode('utf-8')

    # 3. Prepare payload
    payload = {
        "image": base64_image,
        "feature": "currency"
    }

    # 4. Send request
    url = "http://localhost:8000/analyze"
    print(f"Sending Currency Analyze request to {url} using {image_path}...")
    
    max_retries = 5
    for i in range(max_retries):
        try:
            response = requests.post(url, json=payload)
            
            if response.status_code == 200:
                print("\nSuccess! Analysis Result:")
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
    test_currency()
