"""
OpenAI provider implementation for the model provider system
"""

import asyncio
import base64
import logging
import os
import time
from typing import Dict, Any, Optional

import openai
from pydantic import BaseModel

from app.model_providers import ModelProvider, ImageModel, AudioModel
from app.settings import AppConfig
from app.storage import upload_file_to_s3


class OpenAIProvider(ModelProvider):
    """OpenAI provider for text, image, and audio generation"""
    
    def __init__(self):
        super().__init__("openai")
        self.client = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize OpenAI client with API key"""
        # Check if using dummy AI first
        use_dummy_ai = os.environ.get("USE_DUMMY_AI", "false").lower() in ("true", "1", "yes")
        
        if use_dummy_ai:
            self.client = None
            self.logger.info("OpenAI client initialized in dummy mode")
            return
            
        api_key = AppConfig.get_openai_api_key()
        if not api_key:
            raise ValueError("OpenAI API key not found in environment variables or config")
        
        self.client = openai.AsyncOpenAI(api_key=api_key)
        self.logger.info("OpenAI client initialized")
    
    async def generate_text(self, prompt: str, model: str, response_schema: BaseModel = None, **kwargs) -> Dict[str, Any]:
        """Generate structured text using OpenAI models"""
        start_time = time.time()
        self._log_request("text_generation", model, prompt_length=len(prompt))
        
        try:
            # Check if using dummy AI mode
            if self.client is None:
                duration = time.time() - start_time
                if response_schema:
                    # Create a dummy StoryStructure for testing
                    dummy_story = response_schema(
                        title="Dummy Story Title",
                        characters=[{"name": "Dummy Character", "description": "A test character"}],
                        pages=[{
                            "text": "This is a dummy story page for testing.",
                            "imagePrompt": "A dummy image prompt"
                        }]
                    )
                    return {
                        "text": '{"title": "Dummy Story Title", "characters": ["Dummy Character"], "pages": [{"page_number": 1, "text": "This is a dummy story page for testing.", "image_prompt": "A dummy image prompt"}]}',
                        "model": model,
                        "usage": {"prompt_tokens": 10, "completion_tokens": 50, "total_tokens": 60},
                        "duration": duration,
                        "structured_output": dummy_story
                    }
                else:
                    return {
                        "text": "This is a dummy response for testing purposes.",
                        "model": model,
                        "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
                        "duration": duration
                    }
            # Prepare request parameters
            request_params = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": kwargs.get("temperature", 0.7),
            }
            
            # Add structured output if schema provided
            if response_schema:
                # Get the JSON schema and ensure additionalProperties is set to false
                schema = response_schema.model_json_schema()
                self._ensure_additional_properties_false(schema)
                
                request_params["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "story_response",
                        "schema": schema,
                        "strict": True
                    }
                }
            
            # Make API call
            response = await self.client.chat.completions.create(**request_params)
            
            # Calculate duration after API call
            duration = time.time() - start_time
            
            # Extract content
            content = response.choices[0].message.content
            
            # Parse structured output if schema was provided
            if response_schema and content:
                try:
                    import json
                    parsed_content = json.loads(content)
                    # Convert dictionary to Pydantic object for consistency with Gemini provider
                    structured_output = response_schema(**parsed_content)
                    result = {
                        "text": content,
                        "model": model,
                        "usage": {
                            "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                            "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                            "total_tokens": response.usage.total_tokens if response.usage else 0,
                        },
                        "duration": duration,
                        "structured_output": structured_output
                    }
                except json.JSONDecodeError as e:
                    self.logger.warning(f"Failed to parse structured output: {e}")
                    result = {
                        "text": content,
                        "model": model,
                        "usage": {
                            "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                            "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                            "total_tokens": response.usage.total_tokens if response.usage else 0,
                        },
                        "duration": duration,
                        "parse_error": str(e)
                    }
            else:
                result = {
                    "text": content,
                    "model": model,
                    "usage": {
                        "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                        "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                        "total_tokens": response.usage.total_tokens if response.usage else 0,
                    },
                    "duration": duration
                }
            
            self._log_success("text_generation", model, duration)
            
            return result
            
        except Exception as e:
            duration = time.time() - start_time
            self._log_error("text_generation", model, e, duration)
            raise
    
    async def generate_image(self, prompt: str, model: str, **kwargs) -> str:
        """Generate image using OpenAI models and return S3 URL"""
        start_time = time.time()
        self._log_request("image_generation", model, prompt_length=len(prompt))
        
        try:
            # Model-specific configurations
            config = self._get_image_config(model)
            
            # Prepare request parameters
            request_params = {
                "model": model,
                "prompt": prompt,
                "n": 1,
                "response_format": "b64_json",  # Always use b64_json for consistency
                "quality": config["quality"],
                "size": config["size"]
            }
            
            # Make API call
            response = await self.client.images.generate(**request_params)
            
            # Extract base64 image data
            image_data = response.data[0].b64_json
            image_bytes = base64.b64decode(image_data)
            
            # Upload to S3
            s3_url = upload_file_to_s3(
                image_bytes,
                file_type="images",
                extension="png"
            )
            
            duration = time.time() - start_time
            self._log_success("image_generation", model, duration)
            
            return s3_url
            
        except Exception as e:
            duration = time.time() - start_time
            self._log_error("image_generation", model, e, duration)
            raise
    
    async def generate_audio(self, text: str, model: str, voice: str = None, **kwargs) -> str:
        """Generate audio using OpenAI TTS and return S3 URL"""
        start_time = time.time()
        self._log_request("audio_generation", model, text_length=len(text), voice=voice)
        
        try:
            # Use provided voice or default
            voice = voice or "nova"
            
            # Prepare request parameters
            request_params = {
                "model": model,
                "input": text,
                "voice": voice,
                "response_format": "mp3"
            }
            
            # Make API call
            response = await self.client.audio.speech.create(**request_params)
            
            # Get audio bytes
            audio_bytes = response.content
            
            # Upload to S3
            s3_url = upload_file_to_s3(
                audio_bytes,
                file_type="audio",
                extension="mp3"
            )
            
            duration = time.time() - start_time
            self._log_success("audio_generation", model, duration)
            
            return s3_url
            
        except Exception as e:
            duration = time.time() - start_time
            self._log_error("audio_generation", model, e, duration)
            raise
    
    def _ensure_additional_properties_false(self, schema: dict):
        """Recursively ensure additionalProperties is false for all objects in schema"""
        if isinstance(schema, dict):
            # If this is an object type, set additionalProperties to false
            if schema.get("type") == "object":
                schema["additionalProperties"] = False
            
            # Recursively process all nested schemas
            for key, value in schema.items():
                if isinstance(value, dict):
                    self._ensure_additional_properties_false(value)
                elif isinstance(value, list):
                    for item in value:
                        if isinstance(item, dict):
                            self._ensure_additional_properties_false(item)
    
    def _get_image_config(self, model: str) -> Dict[str, str]:
        """Get model-specific image generation configuration"""
        configs = {
            ImageModel.GPT_IMAGE_1: {
                "quality": "medium",
                "size": "1024x1024"
            },
            ImageModel.DALL_E_2: {
                "quality": "standard", 
                "size": "512x512"
            },
            ImageModel.DALL_E_3: {
                "quality": "standard",
                "size": "1024x1024"
            }
        }
        
        return configs.get(model, {
            "quality": "standard",
            "size": "1024x1024"
        }) 