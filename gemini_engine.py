import os
import google.generativeai as genai
from PIL import Image
import io
import logging

logger = logging.getLogger(__name__)

class GeminiEngine:
    def __init__(self):
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            logger.warning("GEMINI_API_KEY environment variable not set. Gemini features will not work.")
            self.model = None
        else:
            genai.configure(api_key=api_key)
            # Use Gemini Flash Latest for best availability
            self.model = genai.GenerativeModel('gemini-flash-latest')
            logger.info("Gemini Engine initialized successfully.")

    def generate_content(self, image_bytes: bytes, prompt: str) -> str:
        if not self.model:
            return "Error: Gemini API key not configured."

        try:
            image = Image.open(io.BytesIO(image_bytes))
            response = self.model.generate_content([prompt, image])
            return response.text
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            return f"Error processing request: {str(e)}"
