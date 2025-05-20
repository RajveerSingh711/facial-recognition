from http.server import BaseHTTPRequestHandler
import json
import io
import base64
import logging
import os
from datetime import datetime

import cv2
import numpy as np
from fer import FER

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

detector = FER()

def preprocess_image(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Resize to 48x48 (FER model's expected input size)
    resized = cv2.resize(gray, (48, 48))
    # Normalize pixel values
    normalized = resized / 255.0
    # Add batch and channel dimensions
    return np.expand_dims(np.expand_dims(normalized, axis=0), axis=-1)

class handler(BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
        return

    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        payload = json.loads(post_data.decode('utf-8'))
        
        try:
            imageByt64 = payload['data']['image'].split(',')[1]
            
            # decode and convert into image
            image = np.fromstring(base64.b64decode(imageByt64), np.uint8)
            image = cv2.imdecode(image, cv2.IMREAD_COLOR)
            
            if image is None:
                self.send_error(400, "Failed to decode image")
                return
            
            # Preprocess image before detection
            processed_image = preprocess_image(image)
            
            # Detect Emotion via Tensorflow model
            prediction = detector.detect_emotions(image)
            if prediction and len(prediction) > 0:
                emotions = prediction[0]['emotions']
                dominant_emotion = max(emotions.items(), key=lambda x: x[1])
                
                response = {
                    "predictions": emotions,
                    "emotion": dominant_emotion[0]
                }
            else:
                response = {
                    "predictions": {
                        "angry": 0,
                        "disgust": 0,
                        "fear": 0,
                        "happy": 0,
                        "sad": 0,
                        "surprise": 0,
                        "neutral": 0
                    },
                    "emotion": "neutral"
                }
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(response).encode())
            
        except Exception as e:
            logger.error(f"Error processing request: {str(e)}")
            self.send_error(500, str(e)) 