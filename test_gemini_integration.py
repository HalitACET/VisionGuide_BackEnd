import unittest
from unittest.mock import MagicMock, patch
import sys
import os
import base64
import json

# Add current directory to path
sys.path.append(os.getcwd())

# Mock heavy dependencies BEFORE importing main
sys.modules['torch'] = MagicMock()
sys.modules['tensorflow'] = MagicMock()
sys.modules['tensorflow_hub'] = MagicMock()
sys.modules['ultralytics'] = MagicMock()

# Mock yolo_engine
mock_yolo = MagicMock()
sys.modules['yolo_engine'] = mock_yolo

# Now we can import main
from main import app, GeminiEngine

from fastapi.testclient import TestClient

class TestGeminiIntegration(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        
    @patch('main.gemini_engine')
    def test_ask_gemini_endpoint(self, mock_gemini_engine):
        # Mock the generate_content method
        mock_gemini_engine.generate_content.return_value = "This is a mocked response from Gemini."
        
        # Create a dummy base64 image
        dummy_image = base64.b64encode(b"fake_image_data").decode('utf-8')
        
        payload = {
            "image": dummy_image,
            "prompt": "What is this?"
        }
        
        response = self.client.post("/ask_gemini", json=payload)
        
        # Verify response
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"answer": "This is a mocked response from Gemini."})
        
        # Verify gemini_engine was called
        mock_gemini_engine.generate_content.assert_called_once()

    @patch('google.generativeai.GenerativeModel')
    @patch('google.generativeai.configure')
    def test_gemini_engine_initialization(self, mock_configure, mock_model):
        # Test with API key
        with patch.dict(os.environ, {"GEMINI_API_KEY": "fake_key"}):
            engine = GeminiEngine()
            mock_configure.assert_called_with(api_key="fake_key")
            mock_model.assert_called_with('gemini-1.5-flash')
            self.assertIsNotNone(engine.model)

    @patch('google.generativeai.GenerativeModel')
    def test_gemini_engine_no_key(self, mock_model):
        # Test without API key
        with patch.dict(os.environ, {}, clear=True):
            engine = GeminiEngine()
            self.assertIsNone(engine.model)

if __name__ == '__main__':
    unittest.main()
