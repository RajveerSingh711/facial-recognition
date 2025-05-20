import json
import io
import base64
import logging
import os
from datetime import datetime

from typing import Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

import cv2
import numpy as np
from fer import FER

from database import store_prediction, get_recent_predictions

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount the static directory for serving images
IMAGES_DIR = os.path.join(os.path.dirname(__file__), 'data', 'images')
os.makedirs(IMAGES_DIR, exist_ok=True)
app.mount("/data", StaticFiles(directory=os.path.join(os.path.dirname(__file__), 'data')), name="data")

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

def save_image(image, emotion):
    """Save the image with timestamp and emotion label."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_{emotion}.jpg"
    filepath = os.path.join(IMAGES_DIR, filename)
    cv2.imwrite(filepath, image)
    # Store only the filename in the database
    return filename

@app.get("/predictions")
async def get_predictions(limit: int = 10, emotion: Optional[str] = None, sort: Optional[str] = None):
    """Get recent predictions from the database with optional filtering and sorting."""
    # Pass the emotion and sort parameters to the database function
    return get_recent_predictions(limit=limit, emotion_filter=emotion, sort_by=sort)

@app.websocket("/")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket connection accepted")
    
    try:
        while True:
            try:
                payload = await websocket.receive_text()
                payload = json.loads(payload)
                imageByt64 = payload['data']['image'].split(',')[1]
                
                # decode and convert into image
                image = np.fromstring(base64.b64decode(imageByt64), np.uint8)
                image = cv2.imdecode(image, cv2.IMREAD_COLOR)
                
                if image is None:
                    logger.warning("Failed to decode image")
                    continue
                
                # Preprocess image before detection
                processed_image = preprocess_image(image)
                
                # Detect Emotion via Tensorflow model
                prediction = detector.detect_emotions(image)
                if prediction and len(prediction) > 0:
                    emotions = prediction[0]['emotions']
                    dominant_emotion = max(emotions.items(), key=lambda x: x[1])
                    
                    # Save image and store prediction with relative path
                    image_filename = save_image(image, dominant_emotion[0])
                    store_prediction(dominant_emotion[0], dominant_emotion[1], image_filename)
                    
                    response = {
                        "predictions": emotions,
                        "emotion": dominant_emotion[0]
                    }
                    await websocket.send_json(response)
                else:
                    await websocket.send_json({
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
                    })
            except WebSocketDisconnect:
                logger.info("WebSocket disconnected")
                break
            except Exception as e:
                logger.error(f"Error processing frame: {str(e)}")
                continue
                
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
    finally:
        try:
            await websocket.close()
        except:
            pass
        logger.info("WebSocket connection closed")