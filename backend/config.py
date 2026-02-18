"""
Configuration management for the music genre classification backend.
"""
import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # API Settings
    app_name: str = "Music Genre Classification API"
    app_version: str = "1.0.0"
    debug: bool = False
    host: str = "0.0.0.0"
    port: int = 8000
    
    # CORS Settings
    cors_origins: list[str] = ["http://localhost:3000", "http://127.0.0.1:3000"]
    
    # Model Settings
    model_checkpoint_path: str = r"C:\SoundModel\artifacts\gtzan_ultimate\best_ultimate.pth"
    device: str = "cuda"  # or "cpu"
    
    # Audio Processing Settings
    sample_rate: int = 22050
    n_fft: int = 2048
    hop_length: int = 512
    n_mels: int = 256  # Match the trained model (was 128, but model expects 256)
    max_audio_duration: float = 30.0  # Process only 30s for faster inference (5s target)
    
    # File Upload Settings
    max_upload_size: int = 50 * 1024 * 1024  # 50 MB
    allowed_extensions: set[str] = {".mp3", ".wav", ".flac", ".ogg", ".m4a"}
    upload_dir: Path = Path("uploads")
    
    # Performance Settings
    max_workers: int = 4
    inference_timeout: int = 30  # seconds
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False
    )
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Create upload directory if it doesn't exist
        self.upload_dir.mkdir(parents=True, exist_ok=True)


# Global settings instance
settings = Settings()


# Genre to emotion mapping
GENRE_TO_EMOTION = {
    "blues": {"emotion": "Melancholic", "description": "Deep, soulful, and introspective", "color": "#4A5899"},
    "classical": {"emotion": "Serene", "description": "Elegant, peaceful, and refined", "color": "#8B7355"},
    "country": {"emotion": "Nostalgic", "description": "Heartfelt, sentimental, and down-to-earth", "color": "#D4A574"},
    "disco": {"emotion": "Euphoric", "description": "Groovy, danceable, and fun", "color": "#FF6B9D"},
    "hiphop": {"emotion": "Confident", "description": "Bold, rhythmic, and expressive", "color": "#FF9F1C"},
    "jazz": {"emotion": "Sophisticated", "description": "Smooth, improvisational, and cool", "color": "#2EC4B6"},
    "metal": {"emotion": "Intense", "description": "Powerful, aggressive, and raw", "color": "#E71D36"},
    "pop": {"emotion": "Joyful", "description": "Upbeat, catchy, and energetic", "color": "#FF006E"},
    "reggae": {"emotion": "Chill", "description": "Laid-back, groovy, and positive", "color": "#06FFA5"},
    "rock": {"emotion": "Energetic", "description": "Dynamic, powerful, and passionate", "color": "#FB5607"}
}
