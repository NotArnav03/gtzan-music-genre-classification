"""
FastAPI application for music genre classification.
"""
import time
import traceback
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from config import settings, GENRE_TO_EMOTION
from schemas import PredictionResponse, ErrorResponse, HealthResponse, ModelInfoResponse
from model_service import ModelService
from audio_processor import AudioProcessor


# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Professional music genre classification API using deep learning",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
model_service: Optional[ModelService] = None
audio_processor: Optional[AudioProcessor] = None


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    global model_service, audio_processor
    
    print("🚀 Starting Music Genre Classification API...")
    print(f"   Version: {settings.app_version}")
    print(f"   Device: {settings.device}")
    
    try:
        # Initialize model service
        model_service = ModelService.get_instance()
        
        # Initialize audio processor with model's parameters
        audio_processor = AudioProcessor(
            sample_rate=settings.sample_rate,
            n_fft=settings.n_fft,
            hop_length=settings.hop_length,
            n_mels=model_service.n_mels  # Use n_mels from loaded model
        )
        
        print("✅ API ready to serve requests!")
    except Exception as e:
        print(f"❌ Failed to initialize services: {str(e)}")
        raise


@app.get("/", response_model=dict)
async def root():
    """Root endpoint."""
    return {
        "message": "Music Genre Classification API",
        "version": settings.app_version,
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        model_loaded=model_service is not None,
        version=settings.app_version
    )


@app.get("/model/info", response_model=ModelInfoResponse)
async def get_model_info():
    """Get model information."""
    if model_service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    
    info = model_service.get_model_info()
    return ModelInfoResponse(**info)


@app.post("/predict", response_model=PredictionResponse)
async def predict_genre(file: UploadFile = File(...)):
    """
    Predict music genre from audio file.
    
    Args:
        file: Audio file (mp3, wav, flac, ogg, m4a)
        
    Returns:
        Prediction results including genre, confidence, and emotion
    """
    start_time = time.time()
    temp_file_path = None
    
    try:
        # Validate file
        if not file.filename:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No filename provided"
            )
        
        # Check file extension
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in settings.allowed_extensions:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported file format. Allowed: {', '.join(settings.allowed_extensions)}"
            )
        
        # Save uploaded file temporarily (optimized I/O)
        temp_file_path = settings.upload_dir / f"temp_{int(time.time() * 1000)}{file_ext}"
        
        # Stream write for better performance
        content = await file.read()
        
        # Check file size
        if len(content) > settings.max_upload_size:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File too large. Max size: {settings.max_upload_size / (1024*1024):.1f} MB"
            )
        
        with open(temp_file_path, "wb") as buffer:
            buffer.write(content)
        
        # Extract features (limited to 30s for faster processing)
        features, duration = audio_processor.extract_features(temp_file_path)
        
        # Pad or truncate to target length (fast operation)
        features = AudioProcessor.pad_or_truncate(
            features,
            model_service.target_frames,
            mode='center'
        )
        
        # Normalize features (only if model was trained with normalization)
        if model_service.use_normalization:
            features = AudioProcessor.normalize_features(
                features,
                model_service.global_mean,
                model_service.global_std
            )
        else:
            print("ℹ️  Skipping normalization (model trained without it)")
        
        # Get prediction
        genre, confidence, all_probs, inference_time = model_service.predict(features)
        
        # Get emotion mapping
        emotion_data = GENRE_TO_EMOTION.get(genre, {
            "emotion": "Unknown",
            "description": "No description available",
            "color": "#808080"
        })
        
        # Calculate total processing time
        total_time = time.time() - start_time
        
        # Sort probabilities by confidence
        sorted_probs = dict(sorted(all_probs.items(), key=lambda x: x[1], reverse=True))
        
        return PredictionResponse(
            success=True,
            genre=genre,
            confidence=confidence,
            emotion=emotion_data["emotion"],
            emotion_description=emotion_data["description"],
            color=emotion_data["color"],
            all_probabilities=sorted_probs,
            processing_time=total_time
        )
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )
    finally:
        # Clean up temporary file
        if temp_file_path and temp_file_path.exists():
            try:
                temp_file_path.unlink()
            except Exception as e:
                print(f"Warning: Failed to delete temp file: {e}")


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler."""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            success=False,
            error=exc.detail,
            detail=str(exc.detail) if exc.detail else None
        ).model_dump()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """General exception handler."""
    print(f"Unhandled exception: {str(exc)}")
    print(traceback.format_exc())
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            success=False,
            error="Internal server error",
            detail=str(exc) if settings.debug else None
        ).model_dump()
    )


if __name__ == "__main__":
    uvicorn.run(
        "api:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug
    )
