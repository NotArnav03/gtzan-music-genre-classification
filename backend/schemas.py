"""
Pydantic schemas for request/response validation.
"""
from typing import Dict, List, Optional
from pydantic import BaseModel, Field, field_validator


class PredictionResponse(BaseModel):
    """Response model for genre prediction."""
    
    success: bool = Field(..., description="Whether the prediction was successful")
    genre: str = Field(..., description="Predicted music genre")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Prediction confidence score")
    emotion: str = Field(..., description="Emotion associated with the genre")
    emotion_description: str = Field(..., description="Detailed description of the emotion")
    color: str = Field(..., description="Color representing the genre")
    all_probabilities: Dict[str, float] = Field(..., description="Probabilities for all genres")
    processing_time: float = Field(..., description="Time taken for inference in seconds")
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "genre": "jazz",
                "confidence": 0.94,
                "emotion": "Sophisticated",
                "emotion_description": "Smooth, improvisational, and cool",
                "color": "#2EC4B6",
                "all_probabilities": {
                    "jazz": 0.94,
                    "blues": 0.03,
                    "classical": 0.02
                },
                "processing_time": 1.23
            }
        }


class ErrorResponse(BaseModel):
    """Response model for errors."""
    
    success: bool = Field(default=False, description="Always false for errors")
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": False,
                "error": "Invalid file format",
                "detail": "Supported formats: mp3, wav, flac, ogg, m4a"
            }
        }


class HealthResponse(BaseModel):
    """Response model for health check."""
    
    status: str = Field(..., description="API status")
    model_loaded: bool = Field(..., description="Whether the ML model is loaded")
    version: str = Field(..., description="API version")
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "model_loaded": True,
                "version": "1.0.0"
            }
        }


class ModelInfoResponse(BaseModel):
    """Response model for model information."""
    
    num_classes: int = Field(..., description="Number of genre classes")
    genres: List[str] = Field(..., description="List of all genre labels")
    model_architecture: str = Field(..., description="Model architecture description")
    target_frames: int = Field(..., description="Target number of time frames")
    n_mels: int = Field(..., description="Number of mel frequency bins")
    
    class Config:
        json_schema_extra = {
            "example": {
                "num_classes": 10,
                "genres": ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"],
                "model_architecture": "CNN + BiLSTM + Transformer + Attention",
                "target_frames": 1292,
                "n_mels": 128
            }
        }
