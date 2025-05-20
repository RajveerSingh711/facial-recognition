import React, { useEffect, useRef, useState } from 'react';
import "./App.css";
import PredictionHistory from './components/PredictionHistory';

function App() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const wsRef = useRef(null);
  const reconnectTimeoutRef = useRef(null);
  const [emotion, setEmotion] = useState('neutral');
  const [predictions, setPredictions] = useState({});
  const [isConnected, setIsConnected] = useState(false);
  const [error, setError] = useState(null);
  const [showHistory, setShowHistory] = useState(false);

  const connectWebSocket = () => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return;

    const ws = new WebSocket('ws://localhost:8000');
    wsRef.current = ws;

    ws.onopen = () => {
      console.log('WebSocket Connected');
      setIsConnected(true);
      setError(null);
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
        reconnectTimeoutRef.current = null;
      }
    };

    ws.onclose = () => {
      console.log('WebSocket Disconnected');
      setIsConnected(false);
      setError('WebSocket disconnected. Attempting to reconnect...');
      reconnectTimeoutRef.current = setTimeout(connectWebSocket, 2000);
    };

    ws.onerror = (error) => {
      console.error('WebSocket Error:', error);
      setError('WebSocket connection error. Attempting to reconnect...');
      setIsConnected(false);
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.error) {
          setError(`Server Error: ${data.error}`);
          console.error('Backend Error:', data.error);
        } else {
          if (error) setError(null);
          setEmotion(data.emotion);
          setPredictions(data.predictions);
        }
      } catch (error) {
        console.error('Error parsing WebSocket message:', error);
        setError('Error processing data from server.');
      }
    };
  };

  useEffect(() => {
    const video = videoRef.current;
    video.width = 640;
    video.height = 480;

    navigator.mediaDevices.getUserMedia({ video: true })
      .then((stream) => {
        video.srcObject = stream;
        video.play();
        connectWebSocket();
      })
      .catch((err) => {
        console.error("Error accessing webcam:", err);
        setError("Error accessing webcam. Please check permissions.");
      });

    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
      if (video.srcObject) {
        video.srcObject.getTracks().forEach(track => track.stop());
      }
    };
  }, []);

  useEffect(() => {
    const sendFrame = () => {
      if (!isConnected || !videoRef.current || !wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return;

      const video = videoRef.current;
      const canvas = canvasRef.current;
      const context = canvas.getContext('2d');

      canvas.width = video.videoWidth || 640;
      canvas.height = video.videoHeight || 480;

      context.drawImage(video, 0, 0, canvas.width, canvas.height);

      const imageData = canvas.toDataURL('image/jpeg', 0.8);
      wsRef.current.send(JSON.stringify({
        data: {
          image: imageData
        }
      }));
    };

    const intervalId = setInterval(sendFrame, 200);

    return () => clearInterval(intervalId);
  }, [isConnected]);

  return (
    <div className="App">
      <header className="App-header">
        <h1>Facial Emotion Recognition</h1>
        {error && <div className="error-message">{error}</div>}
        {!isConnected && !error && (
          <div className="loading-message">Connecting to server...</div>
        )}
        <div className="main-content">
          <div className="video-section">
            <div className="video-container">
              <video ref={videoRef} autoPlay playsInline />
              <canvas ref={canvasRef} width="640" height="480" style={{ display: 'none' }} />
            </div>
            <button 
              className="history-toggle-btn"
              onClick={() => setShowHistory(!showHistory)}
            >
              {showHistory ? 'Hide Prediction History' : 'Show Prediction History'}
            </button>
          </div>
          <div className="emotion-display">
            <h2>Detected Emotion: {emotion}</h2>
            <div className="predictions">
              {Object.entries(predictions).map(([emotion, value]) => (
                <div key={emotion} className="prediction-bar">
                  <span>{emotion}:</span>
                  <div className="bar-container">
                    <div 
                      className="bar" 
                      style={{ 
                        width: `${value * 100}%`,
                        backgroundColor: emotion === 'happy' ? '#4CAF50' : 
                                       emotion === 'sad' ? '#2196F3' : 
                                       emotion === 'angry' ? '#f44336' : 
                                       emotion === 'surprise' ? '#FFC107' : 
                                       emotion === 'fear' ? '#9C27B0' : 
                                       emotion === 'disgust' ? '#795548' : 
                                       '#607D8B'
                      }}
                    />
                  </div>
                  <span>{Math.round(value * 100)}%</span>
                </div>
              ))}
            </div>
          </div>
          {showHistory && <PredictionHistory />}
        </div>
      </header>
    </div>
  );
}

export default App;
