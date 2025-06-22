"""
Gemini provider implementation for the model provider system
"""

import asyncio
import base64
import concurrent.futures
import io
import logging
import os
import time
import wave
from typing import Dict, Any, Optional

from google import genai
from google.genai import types
from pydantic import BaseModel

from app.model_providers import ModelProvider, ImageModel, AudioModel
from app.settings import AppConfig
from app.storage import upload_file_to_s3


class GeminiSafetyException(Exception):
    """Exception raised when Gemini blocks content due to safety filters"""
    def __init__(self, message: str, finish_reason: str = None, blocked_categories: list = None):
        super().__init__(message)
        self.finish_reason = finish_reason
        self.blocked_categories = blocked_categories or []


class GeminiProvider(ModelProvider):
    """Gemini provider for text, image, and audio generation"""
    
    def __init__(self):
        super().__init__("gemini")
        self._initialize_client()
        # Thread pool for parallel execution of sync Gemini calls
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=5, thread_name_prefix="gemini")
    
    def _initialize_client(self):
        """Initialize Gemini client with API key"""
        # Check if using dummy AI first
        use_dummy_ai = os.environ.get("USE_DUMMY_AI", "false").lower() in ("true", "1", "yes")
        
        if use_dummy_ai:
            self.logger.info("Gemini client initialized in dummy mode")
            self.client = None
            return
            
        api_key = AppConfig.get_google_api_key()
        if not api_key:
            raise ValueError("Google API key not found in environment variables or config")
        
        self.client = genai.Client(api_key=api_key)
        self.logger.info("Gemini client initialized with google-genai package")
    
    def _get_api_key(self):
        """Get the Google API key for TTS client"""
        return AppConfig.get_google_api_key()
    
    async def _run_in_executor(self, func, *args, **kwargs):
        """Run a sync function in the thread pool executor for parallel execution"""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, lambda: func(*args, **kwargs))
    
    def _enhance_prompt_for_safety(self, prompt: str) -> str:
        """Enhance prompt to be more likely to pass safety filters"""
        # Add context that this is for creative children's content
        safety_context = (
            "Please create appropriate, child-friendly creative content. "
            "This is for a children's story application that generates educational and entertaining stories. "
        )
        
        # Prepend safety context to the original prompt
        enhanced_prompt = safety_context + prompt
        
        return enhanced_prompt
    
    def _generate_text_sync(self, prompt: str, model: str, response_schema: BaseModel = None, **kwargs) -> Dict[str, Any]:
        """Synchronous text generation - used internally by async wrapper"""
        start_time = time.time()
        self._log_request("text_generation", model, prompt_length=len(prompt))
        
        try:
            # Check if using dummy AI mode
            use_dummy_ai = os.environ.get("USE_DUMMY_AI", "false").lower() in ("true", "1", "yes")
            
            if use_dummy_ai:
                self.logger.info("Using dummy mode for text generation")
                end_time = time.time()
                duration = end_time - start_time
                dummy_text = f"[DUMMY] This is a simulated response for prompt: {prompt[:50]}..."
                self._log_response("text_generation", model, duration, len(dummy_text))
                
                result = {
                    "text": dummy_text,
                    "model": model,
                    "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
                    "duration": duration
                }
                
                # Add dummy structured output if schema provided
                if response_schema:
                    try:
                        # Create a dummy instance of the schema for testing
                        if hasattr(response_schema, '__name__') and 'Story' in response_schema.__name__:
                            # Create dummy story data
                            dummy_structured = response_schema(
                                title="[DUMMY] Test Story",
                                characters=[{"name": "Test Character", "description": "A dummy character for testing"}],
                                pages=[{"text": "This is dummy page text.", "imagePrompt": "A dummy image prompt"}]
                            )
                        else:
                            # Generic dummy object
                            dummy_structured = response_schema()
                        
                        result["structured_output"] = dummy_structured
                        self.logger.info(f"Added dummy structured output: {type(dummy_structured)}")
                    except Exception as e:
                        self.logger.warning(f"Could not create dummy structured output: {e}")
                
                return result
            
            # Enhance prompt for better safety compliance
            enhanced_prompt = self._enhance_prompt_for_safety(prompt)
            self.logger.debug(f"Enhanced prompt length: {len(enhanced_prompt)} (original: {len(prompt)})")
            
            # Build generation config
            generation_config = types.GenerateContentConfig(
                temperature=kwargs.get("temperature", 0.7),
                safety_settings=[
                    types.SafetySetting(
                        category="HARM_CATEGORY_HARASSMENT",
                        threshold="BLOCK_ONLY_HIGH"
                    ),
                    types.SafetySetting(
                        category="HARM_CATEGORY_HATE_SPEECH", 
                        threshold="BLOCK_ONLY_HIGH"
                    ),
                    types.SafetySetting(
                        category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
                        threshold="BLOCK_ONLY_HIGH"
                    ),
                    types.SafetySetting(
                        category="HARM_CATEGORY_DANGEROUS_CONTENT",
                        threshold="BLOCK_ONLY_HIGH"
                    )
                ]
            )
            
            # Add structured output configuration if schema provided
            if response_schema:
                generation_config.response_mime_type = "application/json"
                generation_config.response_schema = response_schema
            
            # Make API call using the new google-genai client
            response = self.client.models.generate_content(
                model=model,
                contents=enhanced_prompt,
                config=generation_config
            )
            
            # Log the entire response for debugging
            if hasattr(response, 'candidates') and response.candidates:
                for i, candidate in enumerate(response.candidates):
                    self.logger.info(f"Candidate {i}: {candidate}")
                    if hasattr(candidate, 'content'):
                        self.logger.info(f"Candidate {i} content: {candidate.content}")
                        if hasattr(candidate.content, 'parts'):
                            self.logger.info(f"Candidate {i} content parts: {candidate.content.parts}")
                    if hasattr(candidate, 'finish_reason'):
                        self.logger.info(f"Candidate {i} finish_reason: {candidate.finish_reason}")
            
            # Check if response was blocked by safety filters
            if not response.candidates:
                # Check prompt feedback for blocking reason
                if hasattr(response, 'prompt_feedback') and response.prompt_feedback.block_reason:
                    block_reason = response.prompt_feedback.block_reason
                    self.logger.warning(f"Prompt blocked by safety filters: {block_reason}")
                    raise GeminiSafetyException(f"Prompt blocked by safety filters: {block_reason}", finish_reason=str(block_reason))
                else:
                    self.logger.warning("No candidates returned in response")
                    raise GeminiSafetyException("No content generated - response blocked or no candidates returned")
            
            # Check if the first candidate was blocked
            candidate = response.candidates[0]
            if hasattr(candidate, 'finish_reason') and candidate.finish_reason:
                # Check for various finish reasons that indicate blocking
                finish_reason = candidate.finish_reason
                
                # Handle numeric finish reasons and string finish reasons
                finish_reason_value = finish_reason if isinstance(finish_reason, (int, str)) else getattr(finish_reason, 'value', finish_reason)
                finish_reason_name = getattr(finish_reason, 'name', str(finish_reason))
                
                # Log the finish reason for debugging
                self.logger.debug(f"Response finish_reason: {finish_reason_name} (value: {finish_reason_value})")
                
                # Handle MAX_TOKENS - this should be very rare now without token limits
                if finish_reason_name == 'MAX_TOKENS':
                    self.logger.info(f"Response reached natural completion length")
                    # Continue to extract content - this is not an error
                
                # Check for actual blocking conditions
                elif finish_reason_name in ['SAFETY', 'RECITATION', 'PROHIBITED_CONTENT']:
                    self.logger.warning(f"Response blocked with finish_reason: {finish_reason_name} (value: {finish_reason_value})")
                    
                    # Provide detailed safety rating information if available
                    safety_info = ""
                    blocked_categories = []
                    if hasattr(candidate, 'safety_ratings') and candidate.safety_ratings:
                        for rating in candidate.safety_ratings:
                            if hasattr(rating, 'blocked') and rating.blocked:
                                category_name = getattr(rating.category, 'name', str(rating.category))
                                probability_name = getattr(rating.probability, 'name', str(rating.probability))
                                blocked_categories.append(f"{category_name}: {probability_name}")
                        if blocked_categories:
                            safety_info = f" (Blocked categories: {', '.join(blocked_categories)})"
                    
                    raise GeminiSafetyException(
                        f"Content blocked by safety filters: {finish_reason_name}{safety_info}",
                        finish_reason=finish_reason_name,
                        blocked_categories=blocked_categories
                    )
            
            # Extract content - prioritize structured output if schema was provided
            content = ""
            structured_output = None
            
            if response_schema:
                # For structured output, use response.parsed to get instantiated objects
                try:
                    structured_output = response.parsed
                    self.logger.debug(f"Successfully extracted structured output using response.parsed: {type(structured_output)}")
                    
                    # Also get the raw text for logging
                    content = response.text
                    self.logger.debug(f"Raw JSON text length: {len(content)} chars")
                    
                except Exception as e:
                    self.logger.warning(f"response.parsed failed ({e}), falling back to text extraction")
                    # Fall back to text extraction
                    content = response.text
            else:
                # For non-structured output, extract text normally
                try:
                    content = response.text
                    self.logger.debug(f"Successfully extracted content using response.text: {len(content)} chars")
                except Exception as e:
                    # If that fails, try to extract from candidates manually
                    self.logger.debug(f"response.text failed ({e}), trying manual extraction")
                    if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
                        for part in response.candidates[0].content.parts:
                            if hasattr(part, 'text') and part.text:
                                content += part.text
                        self.logger.debug(f"Successfully extracted content manually: {len(content)} chars")
                    else:
                        self.logger.error("Failed to extract any content from response")
                        raise Exception("No content could be extracted from the response")
            
            # Calculate timing and log success
            end_time = time.time()
            duration = end_time - start_time
            self._log_response("text_generation", model, duration, len(content))
            
            result = {
                "text": content,
                "model": model,
                "usage": {
                    "prompt_tokens": getattr(response.usage_metadata, 'prompt_token_count', 0) if hasattr(response, 'usage_metadata') else 0,
                    "completion_tokens": getattr(response.usage_metadata, 'candidates_token_count', 0) if hasattr(response, 'usage_metadata') else 0,
                    "total_tokens": getattr(response.usage_metadata, 'total_token_count', 0) if hasattr(response, 'usage_metadata') else 0,
                },
                "duration": duration
            }
            
            # Add structured output if available
            if structured_output is not None:
                result["structured_output"] = structured_output
                self.logger.info(f"Structured output successfully parsed: {type(structured_output)}")
            
            return result
            
        except GeminiSafetyException:
            # Re-raise safety exceptions as-is
            raise
        except Exception as e:
            end_time = time.time()
            duration = end_time - start_time
            self.logger.error(f"Gemini text generation failed: {str(e)}")
            self._log_response("text_generation", model, duration, 0, error=str(e))
            raise
    
    async def generate_text(self, prompt: str, model: str, response_schema: BaseModel = None, **kwargs) -> Dict[str, Any]:
        """Async wrapper for text generation"""
        return await self._run_in_executor(self._generate_text_sync, prompt, model, response_schema, **kwargs)
    
    def _generate_image_sync(self, prompt: str, model: str, **kwargs) -> str:
        """Synchronous image generation - placeholder implementation"""
        start_time = time.time()
        self._log_request("image_generation", model, prompt_length=len(prompt))
        
        try:
            # For now, return a placeholder - image generation needs to be implemented
            # when the google-genai package supports it properly
            self.logger.warning("Gemini image generation not yet implemented with google-genai package")
            
            # Create a placeholder response
            end_time = time.time()
            duration = end_time - start_time
            self._log_response("image_generation", model, duration, 0, error="Not implemented")
            
            # Return a placeholder URL - in production this would be a real generated image
            return "https://via.placeholder.com/512x512.png?text=Gemini+Image+Placeholder"
            
        except Exception as e:
            end_time = time.time()
            duration = end_time - start_time
            self.logger.error(f"Gemini image generation failed: {str(e)}")
            self._log_response("image_generation", model, duration, 0, error=str(e))
            raise
    
    async def generate_image(self, prompt: str, model: str, **kwargs) -> str:
        """Async wrapper for image generation"""
        return await self._run_in_executor(self._generate_image_sync, prompt, model, **kwargs)
    
    def _generate_audio_sync(self, text: str, model: str, voice: str = None, **kwargs) -> str:
        """Synchronous audio generation using new google-genai TTS capabilities"""
        start_time = time.time()
        self._log_request("audio_generation", model, prompt_length=len(text))
        
        try:
            # Check if using dummy AI mode
            use_dummy_ai = os.environ.get("USE_DUMMY_AI", "false").lower() in ("true", "1", "yes")
            
            if use_dummy_ai:
                self.logger.info("Using dummy mode for TTS generation")
                end_time = time.time()
                duration = end_time - start_time
                self._log_response("audio_generation", model, duration, 1000)  # Fake 1KB audio
                return "https://example.com/dummy-audio.mp3"
            # Use hardcoded voice "Leda" as specified in requirements
            voice_name = "Leda"
            
            self.logger.info(f"Generating audio with Gemini TTS model {model}, voice: {voice_name}")
            
            # Build the speech config using the new google-genai types
            speech_config = types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name=voice_name
                    )
                )
            )
            
            # Generate content with TTS
            response = self.client.models.generate_content(
                model=model,
                contents=text,  # Use text directly without prefix
                config=types.GenerateContentConfig(
                    response_modalities=["AUDIO"],
                    speech_config=speech_config
                )
            )
            
            # Extract audio data from response
            if not response.candidates or not response.candidates[0].content.parts:
                raise Exception("No audio data returned from Gemini TTS")
            
            # Get the audio data from the inline_data
            audio_part = response.candidates[0].content.parts[0]
            if not hasattr(audio_part, 'inline_data') or not audio_part.inline_data:
                raise Exception("No inline_data found in TTS response")
            
            # The data should be base64 encoded PCM audio
            audio_data = audio_part.inline_data.data
            if isinstance(audio_data, str):
                # If it's base64 encoded, decode it
                pcm_data = base64.b64decode(audio_data)
            else:
                # If it's already bytes, use directly
                pcm_data = audio_data
            
            self.logger.info(f"Received {len(pcm_data)} bytes of PCM audio data")
            
            # Convert PCM to MP3 using pydub
            mp3_data = self._convert_pcm_to_mp3(pcm_data)
            
            # Upload MP3 to S3
            audio_url = upload_file_to_s3(
                file_data=mp3_data,
                file_type="audio",
                extension="mp3"
            )
            
            # Calculate timing and log success
            end_time = time.time()
            duration = end_time - start_time
            self._log_response("audio_generation", model, duration, len(mp3_data))
            
            self.logger.info(f"Gemini TTS generation completed successfully: {audio_url}")
            return audio_url
            
        except Exception as e:
            end_time = time.time()
            duration = end_time - start_time
            self.logger.error(f"Gemini audio generation failed: {str(e)}")
            self._log_response("audio_generation", model, duration, 0, error=str(e))
            raise
    
    async def generate_audio(self, text: str, model: str, voice: str = None, **kwargs) -> str:
        """Async wrapper for audio generation"""
        return await self._run_in_executor(self._generate_audio_sync, text, model, voice, **kwargs)
    
    def _convert_pcm_to_mp3(self, pcm_data: bytes, channels: int = 1, rate: int = 24000, sample_width: int = 2) -> bytes:
        """Convert PCM audio data to MP3 format using pydub"""
        try:
            from pydub import AudioSegment
            from pydub.utils import which
            import io
            
            # Check if ffmpeg is available
            if not which("ffmpeg"):
                self.logger.warning("ffmpeg not found, using basic conversion")
            
            # Create AudioSegment from raw PCM data
            # Gemini TTS typically outputs 16-bit PCM at 24kHz mono
            audio_segment = AudioSegment(
                data=pcm_data,
                sample_width=sample_width,  # 2 bytes = 16-bit
                frame_rate=rate,  # 24kHz
                channels=channels  # mono
            )
            
            # Convert to MP3
            mp3_buffer = io.BytesIO()
            audio_segment.export(mp3_buffer, format="mp3", bitrate="128k")
            mp3_data = mp3_buffer.getvalue()
            
            self.logger.info(f"Converted PCM ({len(pcm_data)} bytes) to MP3 ({len(mp3_data)} bytes)")
            return mp3_data
            
        except ImportError as e:
            self.logger.error(f"pydub not available for audio conversion: {e}")
            raise Exception("Audio conversion requires pydub library")
        except Exception as e:
            self.logger.error(f"Failed to convert PCM to MP3: {e}")
            raise Exception(f"Audio conversion failed: {str(e)}")
    
    def __del__(self):
        """Cleanup thread pool on deletion"""
        if hasattr(self, '_executor'):
            self._executor.shutdown(wait=False) 