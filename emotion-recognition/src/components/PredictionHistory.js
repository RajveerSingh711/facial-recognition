import React, { useState, useEffect } from 'react';
import './PredictionHistory.css';

const PredictionHistory = () => {
    const [predictions, setPredictions] = useState([]);
    const [loading, setLoading] = useState(true);
    const [selectedEmotion, setSelectedEmotion] = useState(''); // State for emotion filter
    const [selectedSort, setSelectedSort] = useState('timestamp_desc'); // State for sort option
    const emotionsList = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']; // List of possible emotions

    useEffect(() => {
        const fetchPredictions = async () => {
            setLoading(true);
            try {
                // Construct the URL with filter and sort parameters
                const queryParams = new URLSearchParams();
                queryParams.append('limit', 20); // Keep a limit
                if (selectedEmotion) {
                    queryParams.append('emotion', selectedEmotion);
                }
                if (selectedSort) {
                    queryParams.append('sort', selectedSort);
                }
                const url = `http://localhost:8000/predictions?${queryParams.toString()}`;
                
                const response = await fetch(url);
                const data = await response.json();
                setPredictions(data);
                setLoading(false);
            } catch (error) {
                console.error('Error fetching predictions:', error);
                setLoading(false);
            }
        };

        fetchPredictions();
        // Refresh predictions when filter or sort changes, or periodically
        const interval = setInterval(fetchPredictions, 5000); // Keep periodic refresh
        
        // Cleanup function to clear interval
        return () => clearInterval(interval);
        
    }, [selectedEmotion, selectedSort]); // Add dependencies

    // Function to handle emotion filter change
    const handleEmotionChange = (event) => {
        setSelectedEmotion(event.target.value);
    };

    // Function to handle sort option change
    const handleSortChange = (event) => {
        setSelectedSort(event.target.value);
    };

    if (loading) {
        return <div className="prediction-history loading">Loading prediction history...</div>;
    }

    return (
        <div className="prediction-history">
            <h2>Prediction History</h2>
            <div className="filter-sort-controls">
                {/* Emotion Filter */}
                <label htmlFor="emotion-filter">Filter by Emotion:</label>
                <select id="emotion-filter" value={selectedEmotion} onChange={handleEmotionChange}>
                    <option value="">All Emotions</option>
                    {emotionsList.map(emotion => (
                        <option key={emotion} value={emotion}>{emotion.charAt(0).toUpperCase() + emotion.slice(1)}</option>
                    ))}
                </select>
                
                {/* Sort Option */}
                <label htmlFor="sort-by">Sort by:</label>
                <select id="sort-by" value={selectedSort} onChange={handleSortChange}>
                    <option value="timestamp_desc">Newest First (Timestamp)</option>
                    <option value="timestamp_asc">Oldest First (Timestamp)</option>
                    <option value="confidence_desc">Confidence (High to Low)</option>
                    <option value="confidence_asc">Confidence (Low to High)</option>
                </select>
            </div>
            <div className="predictions-list">
                {predictions.length > 0 ? (
                    predictions.map((prediction, index) => (
                        <div key={index} className="prediction-item">
                            <div className="prediction-info">
                                <span className="emotion">{prediction.emotion}</span>
                                <span className="confidence">
                                    {(prediction.confidence * 100).toFixed(1)}%
                                </span>
                                <span className="timestamp">
                                    {new Date(prediction.timestamp).toLocaleString()}
                                </span>
                            </div>
                            {prediction.image_path && (
                                <img
                                    src={`http://localhost:8000/data/images/${prediction.image_path}`}
                                    alt={`${prediction.emotion} face`}
                                    className="prediction-image"
                                />
                            )}
                        </div>
                    ))
                ) : (
                    <p>No predictions found for the selected criteria.</p>
                )}
            </div>
        </div>
    );
};

export default PredictionHistory; 