import os
import google.generativeai as genai
from PIL import Image
import io
import base64

def verify_api_key():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("No API key found.")
        return

    print(f"Testing with API key: {api_key[:5]}...{api_key[-5:]}")
    
    try:
        genai.configure(api_key=api_key)
        
        print("Listing available models...")
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                print(m.name)
        
        # Try with a fallback model if flash fails
        model_name = 'gemini-flash-latest' # Updated to available model
        print(f"Attempting with {model_name}...")
        model = genai.GenerativeModel(model_name)
        response = model.generate_content("Hello")
        print(f"Success with {model_name}: {response.text}")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    verify_api_key()
