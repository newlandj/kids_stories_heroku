"""
Model provider system for supporting multiple AI providers (OpenAI, Gemini)
Allows mix-and-match of text, image, and audio models from different providers.
"""

import asyncio
import base64
import logging
import os
from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, Any, Optional, Union
import time

# Configure logging
logger = logging.getLogger("kids-story-app")


class TextModel(str, Enum):
    """Available text generation models"""
    # OpenAI Models
    GPT_4O = "gpt-4o"
    GPT_4O_MINI = "gpt-4o-mini" 
    GPT_4_1 = "gpt-4.1"
    GPT_4_1_MINI = "gpt-4.1-mini"
    O3 = "o3"
    
    # Gemini Models
    GEMINI_2_5_FLASH = "gemini-2.5-flash"
    GEMINI_2_5_FLASH_LITE = "gemini-2.5-flash-lite-preview-06-17"
    GEMINI_2_5_PRO = "gemini-2.5-pro"


class ImageModel(str, Enum):
    """Available image generation models"""
    # OpenAI Models
    GPT_IMAGE_1 = "gpt-image-1"
    DALL_E_2 = "dall-e-2"
    DALL_E_3 = "dall-e-3"
    
    # Gemini Models
    GEMINI_2_0_FLASH_IMAGE = "gemini-2.0-flash-preview-image-generation"


class AudioModel(str, Enum):
    """Available audio generation models"""
    # OpenAI Models
    TTS_1 = "tts-1"
    TTS_1_HD = "tts-1-hd"
    
    # Gemini Models
    GEMINI_2_5_FLASH_TTS = "gemini-2.5-flash-preview-tts"


class ModelConfig:
    """Model configuration and display name mappings"""
    
    # Display name mappings
    TEXT_MODEL_NAMES = {
        TextModel.GPT_4O: "GPT-4o",
        TextModel.GPT_4O_MINI: "GPT-4o-mini",
        TextModel.GPT_4_1: "GPT-4.1", 
        TextModel.GPT_4_1_MINI: "GPT-4.1-mini",
        TextModel.O3: "o3",
        TextModel.GEMINI_2_5_FLASH: "Gemini 2.5 Flash",
        TextModel.GEMINI_2_5_FLASH_LITE: "Gemini 2.5 Flash - Lite",
        TextModel.GEMINI_2_5_PRO: "Gemini 2.5 Pro",
    }
    
    # Model-specific parameter configurations
    TEXT_MODEL_CONFIGS = {
        # Standard OpenAI models
        TextModel.GPT_4O: {
            "temperature": 0.9,
            "max_tokens": None,  # Let model decide
            "top_p": 1.0,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
        },
        TextModel.GPT_4O_MINI: {
            "temperature": 0.9,
            "max_tokens": None,
            "top_p": 1.0,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
        },
        TextModel.GPT_4_1: {
            "temperature": 0.9,
            "max_tokens": None,
            "top_p": 1.0,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
        },
        TextModel.GPT_4_1_MINI: {
            "temperature": 0.9,
            "max_tokens": None,
            "top_p": 1.0,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
        },
        # Reasoning models (O3 series) - more restrictive parameters
        TextModel.O3: {
            "temperature": 1.0,  # O3 only supports default temperature
            "max_tokens": None,
            # O3 doesn't support top_p, frequency_penalty, presence_penalty
        },
        # Gemini models
        TextModel.GEMINI_2_5_FLASH: {
            "temperature": 0.9,
            "max_output_tokens": None,
            "top_p": 0.95,
            "top_k": 40,
        },
        TextModel.GEMINI_2_5_FLASH_LITE: {
            "temperature": 0.9,
            "max_output_tokens": None,
            "top_p": 0.95,
            "top_k": 40,
        },
        TextModel.GEMINI_2_5_PRO: {
            "temperature": 0.9,
            "max_output_tokens": None,
            "top_p": 0.95,
            "top_k": 40,
        },
    }
    
    # Image model configurations
    IMAGE_MODEL_CONFIGS = {
        ImageModel.DALL_E_2: {
            "size": "512x512",
            # Quality and style are not supported for DALL-E 2
            # "quality": "standard",
            # "style": "natural",
        },
        ImageModel.DALL_E_3: {
            "size": "1024x1024", # options are 1024x1024, 1792x1024, and 1024x1792
            "quality": "standard", #options are hd and standard
            "style": "vivid", #options are natural and vivid
        },
        ImageModel.GPT_IMAGE_1: {
            "size": "512x512",
            "quality": "standard", 
        },
        ImageModel.GEMINI_2_0_FLASH_IMAGE: {
            # Gemini-specific image parameters
        },
    }
    
    # Audio model configurations
    AUDIO_MODEL_CONFIGS = {
        AudioModel.TTS_1: {
            "voice": "nova",
            "speed": 1.0,
            "response_format": "mp3",
        },
        AudioModel.TTS_1_HD: {
            "voice": "nova", 
            "speed": 1.0,
            "response_format": "mp3",
        },
        AudioModel.GEMINI_2_5_FLASH_TTS: {
            "voice_name": "Leda",
            # Gemini-specific audio parameters
        },
    }
    
    IMAGE_MODEL_NAMES = {
        ImageModel.GPT_IMAGE_1: "GPT-Image-1",
        ImageModel.DALL_E_2: "DALL-E-2",
        ImageModel.DALL_E_3: "DALL-E-3",
        ImageModel.GEMINI_2_0_FLASH_IMAGE: "Gemini 2.0 Flash",
    }
    
    AUDIO_MODEL_NAMES = {
        AudioModel.TTS_1: "TTS-1",
        AudioModel.TTS_1_HD: "TTS-1-HD",
        AudioModel.GEMINI_2_5_FLASH_TTS: "Gemini 2.5 Flash",
    }
    
    # Provider mappings
    TEXT_PROVIDERS = {
        TextModel.GPT_4O: "openai",
        TextModel.GPT_4O_MINI: "openai",
        TextModel.GPT_4_1: "openai",
        TextModel.GPT_4_1_MINI: "openai",
        TextModel.O3: "openai",
        TextModel.GEMINI_2_5_FLASH: "gemini",
        TextModel.GEMINI_2_5_FLASH_LITE: "gemini",
        TextModel.GEMINI_2_5_PRO: "gemini",
    }
    
    IMAGE_PROVIDERS = {
        ImageModel.GPT_IMAGE_1: "openai",
        ImageModel.DALL_E_2: "openai", 
        ImageModel.DALL_E_3: "openai",
        ImageModel.GEMINI_2_0_FLASH_IMAGE: "gemini",
    }
    
    AUDIO_PROVIDERS = {
        AudioModel.TTS_1: "openai",
        AudioModel.TTS_1_HD: "openai",
        AudioModel.GEMINI_2_5_FLASH_TTS: "gemini",
    }
    
    # Default models (as specified in requirements)
    DEFAULT_TEXT_MODEL = TextModel.O3  # Testing O3 model
    DEFAULT_IMAGE_MODEL = ImageModel.DALL_E_2
    DEFAULT_AUDIO_MODEL = AudioModel.TTS_1
    
    @classmethod
    def get_text_model_config(cls, model: TextModel) -> Dict[str, Any]:
        """Get configuration parameters for a text model"""
        return cls.TEXT_MODEL_CONFIGS.get(model, {})
    
    @classmethod
    def get_image_model_config(cls, model: ImageModel) -> Dict[str, Any]:
        """Get configuration parameters for an image model"""
        return cls.IMAGE_MODEL_CONFIGS.get(model, {})
    
    @classmethod
    def get_audio_model_config(cls, model: AudioModel) -> Dict[str, Any]:
        """Get configuration parameters for an audio model"""
        return cls.AUDIO_MODEL_CONFIGS.get(model, {})


class ModelProvider(ABC):
    """Abstract base class for AI model providers"""
    
    def __init__(self, provider_name: str):
        self.provider_name = provider_name
        self.logger = logging.getLogger(f"kids-story-app.{provider_name}")
    
    @abstractmethod
    async def generate_text(self, prompt: str, model: str, **kwargs) -> Dict[str, Any]:
        """Generate structured text using specified model"""
        pass
    
    @abstractmethod 
    async def generate_image(self, prompt: str, model: str, **kwargs) -> str:
        """Generate image and return S3 URL"""
        pass
    
    @abstractmethod
    async def generate_audio(self, text: str, model: str, voice: str = None, **kwargs) -> str:
        """Generate audio and return S3 URL"""
        pass
    
    def _log_request(self, operation: str, model: str, **kwargs):
        """Log provider request for monitoring"""
        self.logger.info(f"{operation} request: provider={self.provider_name}, model={model}, kwargs={kwargs}")
    
    def _log_response(self, operation: str, model: str, duration: float, output_size: int = 0, error: str = None):
        """Log provider response for monitoring"""
        if error:
            self.logger.error(f"{operation} error: provider={self.provider_name}, model={model}, duration={duration:.2f}s, error={error}")
        else:
            self.logger.info(f"{operation} success: provider={self.provider_name}, model={model}, duration={duration:.2f}s, output_size={output_size}")
    
    def _log_success(self, operation: str, model: str, duration: float):
        """Log successful operation"""
        self.logger.info(f"{operation} success: provider={self.provider_name}, model={model}, duration={duration:.2f}s")
    
    def _log_error(self, operation: str, model: str, error: Exception, duration: float):
        """Log operation error with details"""
        self.logger.error(f"{operation} error: provider={self.provider_name}, model={model}, duration={duration:.2f}s, error={str(error)}")


class ModelProviderFactory:
    """Factory for creating model provider instances"""
    
    _providers: Dict[str, ModelProvider] = {}
    
    @classmethod
    def get_provider(cls, provider_name: str) -> ModelProvider:
        """Get or create provider instance"""
        if provider_name not in cls._providers:
            if provider_name == "openai":
                from app.model_providers_openai import OpenAIProvider
                cls._providers[provider_name] = OpenAIProvider()
            elif provider_name == "gemini":
                from app.model_providers_gemini import GeminiProvider  
                cls._providers[provider_name] = GeminiProvider()
            else:
                raise ValueError(f"Unknown provider: {provider_name}")
        
        return cls._providers[provider_name]
    
    @classmethod
    def get_text_provider(cls, model: TextModel) -> ModelProvider:
        """Get provider for text model"""
        provider_name = ModelConfig.TEXT_PROVIDERS[model]
        return cls.get_provider(provider_name)
    
    @classmethod 
    def get_image_provider(cls, model: ImageModel) -> ModelProvider:
        """Get provider for image model"""
        provider_name = ModelConfig.IMAGE_PROVIDERS[model]
        return cls.get_provider(provider_name)
    
    @classmethod
    def get_audio_provider(cls, model: AudioModel) -> ModelProvider:
        """Get provider for audio model"""
        provider_name = ModelConfig.AUDIO_PROVIDERS[model]
        return cls.get_provider(provider_name)


class ModelPreferences:
    """Container for user model preferences"""
    
    def __init__(
        self,
        text_model: TextModel = ModelConfig.DEFAULT_TEXT_MODEL,
        image_model: ImageModel = ModelConfig.DEFAULT_IMAGE_MODEL, 
        audio_model: AudioModel = ModelConfig.DEFAULT_AUDIO_MODEL
    ):
        self.text_model = text_model
        self.image_model = image_model
        self.audio_model = audio_model
    
    @classmethod
    def from_dict(cls, data: Dict[str, str]) -> 'ModelPreferences':
        """Create preferences from API request data"""
        return cls(
            text_model=TextModel(data.get('text_model', ModelConfig.DEFAULT_TEXT_MODEL)),
            image_model=ImageModel(data.get('image_model', ModelConfig.DEFAULT_IMAGE_MODEL)),
            audio_model=AudioModel(data.get('audio_model', ModelConfig.DEFAULT_AUDIO_MODEL))
        )
    
    def to_dict(self) -> Dict[str, str]:
        """Convert to dictionary for API responses"""
        return {
            'text_model': self.text_model.value,
            'image_model': self.image_model.value,
            'audio_model': self.audio_model.value
        } 