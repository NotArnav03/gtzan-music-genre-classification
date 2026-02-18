"""
Backend initialization file.
"""
from .api import app
from .model_service import ModelService
from .audio_processor import AudioProcessor

__all__ = ['app', 'ModelService', 'AudioProcessor']
