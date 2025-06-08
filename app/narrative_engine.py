"""
AI pipeline logic:
- USE_DUMMY_AI=true: Use dummy images/audio (for fast local/dev testing, no API calls)
"""
import os
import logging
import random
from typing import List, Optional, Dict
import time
import asyncio
import json
import uuid
import base64

# For API operations
import openai
from openai import OpenAI, AsyncOpenAI
import httpx

# For file operations
import io
import tempfile
import boto3
from PIL import Image
from urllib.parse import urlparse
import requests

# Local imports
from app.settings import AppConfig
from app.storage import upload_file_to_s3
from app.utils import log_memory_usage
from app.translation_service import TranslationService
from app.models import SupportedLanguage
from app.content_safety import ContentScreener
from app.sample_stories import get_sample_story, SAMPLE_STORIES

# Configure logging
logger = logging.getLogger("kids-story-lambda")

class FableFactory:
    
    def __init__(self):
        # Check if using dummy AI first
        if os.environ.get("USE_DUMMY_AI", "false").lower() in ("true", "1", "yes"):
            self.client = None  # No client needed for dummy mode
        else:
            api_key = self._get_openai_api_key()
            if not api_key:
                raise ValueError("OpenAI API key not found.")
            self.client = OpenAI(api_key=api_key,
                                 http_client=httpx.Client(timeout=60.0, follow_redirects=True))
        # Content safety
        self.content_screener = ContentScreener()
        # Determine environment
        self._environment = os.environ.get("DEPLOYMENT_STAGE", "dev")
        # Narrator voices
        self._narrator_voices = {
            "default": "nova",
            "adventurous": "onyx",
            "gentle": "shimmer",
            "wise": "echo"
        }
        # Define difficulty level guidance based on Flesch-Kincaid scale
        self.difficulty_guidance = {
            0: {
                "age": "3-4 years",
                "description": "Pre-reading", 
                "sentence_length": "Very short sentences (3-5 words)",
                "vocabulary": "Simple 1-syllable words, basic concepts",
                "concepts": "Very basic concepts, everyday objects",
                "target_words": 150,  # Read-aloud speed ~150 wpm, 5 min = 750 words, but shorter for attention span
                "reading_speed": "Read-aloud by parent (~150 wpm)"
            },
            1: {
                "age": "5 years",
                "description": "Beginning sounds",
                "sentence_length": "Short sentences (4-8 words)",
                "vocabulary": "Mostly 1-2 syllable words, simple stories",
                "concepts": "Simple actions and emotions",
                "target_words": 200,  # ~40 wpm × 5 min = 200 words
                "reading_speed": "Beginning reader (~40 wpm)"
            },
            2: {
                "age": "6 years", 
                "description": "Early reading",
                "sentence_length": "Medium sentences (6-12 words)",
                "vocabulary": "Mix of 1-3 syllable words, simple plot",
                "concepts": "Basic storylines, cause and effect",
                "target_words": 400,  # ~80 wpm × 5 min = 400 words
                "reading_speed": "Early reader (~80 wpm)"
            },
            3: {
                "age": "7 years",
                "description": "Developing reading",
                "sentence_length": "Longer sentences (8-15 words)",
                "vocabulary": "Some complex words, more detailed descriptions",
                "concepts": "Character emotions, simple problem-solving",
                "target_words": 575,  # ~115 wpm × 5 min = 575 words
                "reading_speed": "Developing reader (~115 wpm)"
            },
            4: {
                "age": "8 years",
                "description": "Fluent reading", 
                "sentence_length": "Varied sentence length (10-20 words)",
                "vocabulary": "Broader vocabulary, more sophisticated words",
                "concepts": "Complex concepts, detailed storytelling",
                "target_words": 690,  # ~138 wpm × 5 min = 690 words
                "reading_speed": "Fluent reader (~138 wpm)"
            },
            5: {
                "age": "9 years",
                "description": "Advanced reading",
                "sentence_length": "Longer sentences (12-25 words)",
                "vocabulary": "Advanced vocabulary, scientific and technical terms",
                "concepts": "Abstract thinking, cause and effect relationships",
                "target_words": 790,  # ~158 wpm × 5 min = 790 words
                "reading_speed": "Advanced reader (~158 wpm)"
            },
            6: {
                "age": "10 years", 
                "description": "Complex stories",
                "sentence_length": "Complex sentences (15-30 words)",
                "vocabulary": "Sophisticated vocabulary, descriptive language",
                "concepts": "Multiple plot elements, character development",
                "target_words": 865,  # ~173 wpm × 5 min = 865 words
                "reading_speed": "Complex reader (~173 wpm)"
            },
            7: {
                "age": "11 years",
                "description": "Pre-teen literature",
                "sentence_length": "Advanced sentence structure (20-35 words)",
                "vocabulary": "Academic vocabulary, subject-specific terms",
                "concepts": "Real-world issues, moral dilemmas, advanced themes",
                "target_words": 925,  # ~185 wpm × 5 min = 925 words
                "reading_speed": "Pre-teen reader (~185 wpm)"
            },
            8: {
                "age": "12 years",
                "description": "Middle grade literature",
                "sentence_length": "Complex academic language (25-40 words)",
                "vocabulary": "Technical vocabulary, nuanced language",
                "concepts": "Abstract concepts, detailed explanations, advanced reasoning",
                "target_words": 975,  # ~195 wpm × 5 min = 975 words
                "reading_speed": "Middle grade reader (~195 wpm)"
            }
        }

    async def generate_story_package(self, prompt: str, difficulty_level: int = None) -> dict:
        log_memory_usage("narrative_engine.FableFactory.generate_story_package: start")
        """
        Orchestrates the generation of children's story elements using AI (structured JSON pipeline)
        """
        start_time = time.monotonic()
        story = await self.weave_narrative(prompt, difficulty_level)  # story is now a dict with title, characters, pages
        log_memory_usage("narrative_engine.FableFactory.generate_story_package: after weave_narrative")
        pages = story.get("pages", [])
        visual_elements = []
        audio_narration = []
        print(story)
        print("\nget characters", story.get("characters", []))
        print("for loops: \n")
        for idx, page in enumerate(pages):
            c = story.get("characters", [])
            thing = ", ".join(f"{c['name']}: {c['description']}" for c in c)
            print(thing)

        # Parallelize illustration and narration generation for all pages
        # Prepend character descriptions to each imagePrompt
        character_descriptions = ", ".join([f"{c['name']}: {c['description']}" for c in story.get("characters", [])])
        image_tasks = [
            self._generate_single_illustration(f"{character_descriptions}. {page['imagePrompt']}", idx, art_direction=None)
            for idx, page in enumerate(pages)
        ]
        # Select voice once based on the first page's text (if available)
        voice = self._select_voice_for_story(pages[0]["text"] if pages else "")
        audio_tasks = [
            self._generate_single_narration(page["text"], idx, voice)
            for idx, page in enumerate(pages)
        ]
        # Run all tasks in parallel
        visual_elements, audio_narration = await asyncio.gather(
            asyncio.gather(*image_tasks),
            asyncio.gather(*audio_tasks)
        )
        log_memory_usage("narrative_engine.FableFactory.generate_story_package: after images/audio")
        # Wire imageUrl and audioUrl into each page
        for idx, page in enumerate(pages):
            if idx < len(visual_elements):
                page["imageUrl"] = visual_elements[idx]
            # Find matching audio narration by index
            audio = next((a for a in audio_narration if a.get("page_index") == idx), None)
            if audio:
                page["audioUrl"] = audio["audio_url"]
        log_memory_usage("narrative_engine.FableFactory.generate_story_package: after wiring URLs")
        word_count = sum(len(page["text"].split()) for page in pages)
        illustration_count = len(visual_elements)
        page_count = len(pages)
        elapsed = time.monotonic() - start_time
        logger.info(f"Story generation complete in {elapsed:.2f} seconds. Title: {story.get('title')}, {page_count} pages, {illustration_count} images, {word_count} words.")
        log_memory_usage("narrative_engine.FableFactory.generate_story_package: end")
        return {
            "title": story.get("title"),
            "characters": story.get("characters"),
            "pages": pages,
            "visual_elements": visual_elements,
            "audio_narration": audio_narration,
            "word_count": word_count,
            "illustration_count": illustration_count,
            "page_count": page_count
        }
    """Orchestrates the generation of children's story elements using AI"""
        
    def _get_openai_api_key(self) -> Optional[str]:
        return os.environ.get("OPENAI_API_KEY")

    async def weave_narrative(self, prompt: str, difficulty_level: int = None) -> dict:
        log_memory_usage("narrative_engine.FableFactory.weave_narrative: start")
        if os.environ.get("USE_DUMMY_AI", "false").lower() in ("true", "1", "yes"):
            # Minimal dummy response in new structure
            return {
                "title": "The Brave Child's Adventure",
                "characters": [
                    {"name": "Jamie", "description": "a brave 8-year-old with curly brown hair and a blue backpack"}
                ],
                "pages": [
                    {"text": "Jamie found a mysterious map in the attic.", "imagePrompt": "A child holding a map in a dusty attic, sunlight streaming in."},
                    {"text": "Jamie followed the map into the woods, meeting a talking squirrel.", "imagePrompt": "A child and a talking squirrel in a magical forest."},
                    {"text": "Together, they discovered a hidden treasure.", "imagePrompt": "A child and squirrel celebrating next to a treasure chest in the woods."}
                ]
            }
        self.content_screener.validate_prompt(prompt)

        # Define the function schema for OpenAI tool calling
        function_schema = {
            "name": "create_story",
            "description": "Generate a JSON children's story with title, characters, and pages.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "The story title."},
                    "characters": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string", "description": "Character name."},
                                "description": {
                                    "type": "string",
                                    "description": """
                                        Character appearance and traits. The intent of this section is to 
                                        provide a consistent description of the character that can be 
                                        used across all pages of the story. Please be specific and 
                                        make sure to mention the color of the animal, or the skin tone, age,
                                        hair color, etc. if it's a human. 
                                    """
                                }
                            },
                            "required": ["name", "description"]
                        },
                        "description": "List of characters with descriptions."
                    },
                    "pages": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "text": {"type": "string", "description": "Page story text."},
                                "imagePrompt": {
                                    "type": "string",
                                    "description": """
                                        Detailed illustration prompt for this page, including character names and
                                        anything specific they're doing. The detailed descriptions of what the charactesr
                                        look like will be passed into this prompt separately, so focus on the character names
                                        and the actions they're taking. 
                                        """
                                }
                            },
                            "required": ["text", "imagePrompt"]
                        },
                        "description": "Pages of the story."
                    }
                },
                "required": ["title", "characters", "pages"]
            }
        }

        # Build system prompt with difficulty level guidance
        base_prompt = (
            "You are a children's story writer creating engaging, age-appropriate stories for children. "
            "Create stories with 3-4 pages, each with a vivid scene that can be illustrated. For each story, create detailed character descriptions that should remain consistent throughout the story. For each page, provide a detailed image description that maintains character consistency. "
            "IMPORTANT: Each story MUST include a creative, appropriate book title, and the title MUST be returned as the 'title' field in the structured JSON output."
        )
        
        system_prompt = base_prompt
        if difficulty_level is not None:
            difficulty_guidance = self._get_difficulty_guidance(difficulty_level)
            system_prompt = f"{base_prompt}\n\n{difficulty_guidance}"
        user_message = f"Create a children's story about: {prompt}. Please ensure your response includes a creative, appropriate title for the book, returned as the 'title' field in the structured JSON output."

        attempts, backoff = 3, 1
        for i in range(attempts):
            try:
                resp = await asyncio.to_thread(
                    self.client.chat.completions.create,
                    model="gpt-4o",
                    messages=[{"role": "system", "content": system_prompt},
                              {"role": "user", "content": user_message}],
                    temperature=0.7 + i*0.1,
                    max_tokens=1800,
                    presence_penalty=0.6,
                    frequency_penalty=0.5,
                    tools=[{"type": "function", "function": function_schema}],
                    tool_choice={"type": "function", "function": {"name": "create_story"}}
                )
                tool_calls = resp.choices[0].message.tool_calls
                if tool_calls and len(tool_calls) > 0:
                    arguments = tool_calls[0].function.arguments
                    # arguments is a JSON string, parse it
                    story_json = json.loads(arguments)
                    return story_json
                else:
                    logger.error("No tool_calls found in OpenAI response.")
            except Exception as e:
                logger.error(f"weave_narrative error {i}: {e}")
                if i < attempts-1:
                    await asyncio.sleep(backoff); backoff *= 2
        # Fallback
        return self._generate_fallback_story(prompt)
    
    def _get_difficulty_guidance(self, difficulty_level: int) -> str:
        """Generate difficulty-specific guidance for the LLM based on Flesch-Kincaid level."""
        
        level_info = self.difficulty_guidance.get(difficulty_level, self.difficulty_guidance[2])  # Default to level 2
        sample_story = get_sample_story(difficulty_level)
        
        return f"""
IMPORTANT READING LEVEL REQUIREMENTS:
You must write for Flesch-Kincaid Grade Level {difficulty_level} (Age {level_info['age']} - {level_info['description']}).

TARGET STORY LENGTH: {level_info['target_words']} words (for 5-minute reading at {level_info['reading_speed']})

SPECIFIC WRITING GUIDELINES:
- {level_info['sentence_length']}
- {level_info['vocabulary']}
- {level_info['concepts']}

TARGET FLESCH-KINCAID GRADE LEVEL: {difficulty_level}

EXAMPLE OF APPROPRIATE COMPLEXITY FOR LEVEL {difficulty_level}:
Title: "{sample_story['title']}"
Text: "{sample_story['text']}"

Your story should match this level of complexity. Use the Flesch-Kincaid Grade Level formula which considers:
1. Average sentence length (total words ÷ total sentences)
2. Average syllables per word (total syllables ÷ total words)

Keep sentences and vocabulary appropriate for the target age group. Make sure your story would score close to Grade Level {difficulty_level} on the Flesch-Kincaid scale.
"""
    
    async def generate_story_with_readability_feedback(self, prompt: str, target_level: int, current_score: float, previous_story: dict) -> dict:
        """Generate a new story attempt with readability feedback from the previous attempt."""
        log_memory_usage("narrative_engine.FableFactory.generate_story_with_readability_feedback: start")
        
        if os.environ.get("USE_DUMMY_AI", "false").lower() in ("true", "1", "yes"):
            # Return dummy response with image and audio URLs for testing
            return {
                "title": "The Brave Child's Adventure (Retry)",
                "characters": [
                    {"name": "Jamie", "description": "a brave 8-year-old with curly brown hair and a blue backpack"}
                ],
                "pages": [
                    {"text": "Jamie found a map.", "imagePrompt": "A child holding a map.", "imageUrl": "https://dummy-image-1.png", "audioUrl": "https://dummy-audio-1.mp3"},
                    {"text": "Jamie went to the woods.", "imagePrompt": "A child walking into a forest.", "imageUrl": "https://dummy-image-2.png", "audioUrl": "https://dummy-audio-2.mp3"},
                    {"text": "Jamie found treasure.", "imagePrompt": "A child celebrating next to a treasure chest.", "imageUrl": "https://dummy-image-3.png", "audioUrl": "https://dummy-audio-3.mp3"}
                ]
            }
        
        self.content_screener.validate_prompt(prompt)
        
        # Combine all previous story text for analysis
        previous_text = " ".join([page.get("text", "") for page in previous_story.get("pages", [])])
        
        # Create feedback-based prompt
        feedback_prompt = self._create_readability_feedback_prompt(
            prompt, target_level, current_score, previous_text
        )
        
        # Use the same function schema as regular story generation
        function_schema = {
            "name": "create_story",
            "description": "Generate a JSON children's story with title, characters, and pages.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "The story title."},
                    "characters": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string", "description": "Character name."},
                                "description": {
                                    "type": "string",
                                    "description": "Character appearance and traits for consistent illustration."
                                }
                            },
                            "required": ["name", "description"]
                        },
                        "description": "List of characters with descriptions."
                    },
                    "pages": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "text": {"type": "string", "description": "Page story text."},
                                "imagePrompt": {
                                    "type": "string",
                                    "description": "Detailed illustration prompt for this page."
                                }
                            },
                            "required": ["text", "imagePrompt"]
                        },
                        "description": "Pages of the story."
                    }
                },
                "required": ["title", "characters", "pages"]
            }
        }
        
        # If not using dummy AI, try OpenAI API
        if self.client is not None:
            attempts, backoff = 3, 1
            for i in range(attempts):
                try:
                    resp = await asyncio.to_thread(
                        self.client.chat.completions.create,
                        model="gpt-4o",
                        messages=[{"role": "system", "content": feedback_prompt},
                                  {"role": "user", "content": f"Rewrite the story about: {prompt}. Adjust the complexity to better match Grade Level {target_level}."}],
                        temperature=0.7 + i*0.1,
                        max_tokens=1800,
                        presence_penalty=0.6,
                        frequency_penalty=0.5,
                        tools=[{"type": "function", "function": function_schema}],
                        tool_choice={"type": "function", "function": {"name": "create_story"}}
                    )
                    tool_calls = resp.choices[0].message.tool_calls
                    if tool_calls and len(tool_calls) > 0:
                        arguments = tool_calls[0].function.arguments
                        story_json = json.loads(arguments)
                        # Generate images and audio for the retry story
                        complete_story = await self._generate_retry_story_with_media(story_json)
                        return complete_story
                    else:
                        logger.error("No tool_calls found in readability feedback response.")
                except Exception as e:
                    logger.error(f"generate_story_with_readability_feedback error {i}: {e}")
                    if i < attempts-1:
                        await asyncio.sleep(backoff); backoff *= 2
        
        # If we get here, OpenAI retry failed, fallback to original story
        logger.warning("Readability feedback retry failed, returning original story")
        return previous_story
    
    async def _generate_retry_story_with_media(self, story_data: dict) -> dict:
        """Generate images and audio for a retry story, similar to generate_story_package."""
        log_memory_usage("narrative_engine.FableFactory._generate_retry_story_with_media: start")
        
        pages = story_data.get("pages", [])
        if not pages:
            return story_data
        
        # Generate images and audio using the same logic as generate_story_package
        character_descriptions = ", ".join([f"{c['name']}: {c['description']}" for c in story_data.get("characters", [])])
        image_tasks = [
            self._generate_single_illustration(f"{character_descriptions}. {page['imagePrompt']}", idx, art_direction=None)
            for idx, page in enumerate(pages)
        ]
        
        # Select voice for audio generation
        voice = self._select_voice_for_story(pages[0]["text"] if pages else "")
        audio_tasks = [
            self._generate_single_narration(page["text"], idx, voice)
            for idx, page in enumerate(pages)
        ]
        
        # Generate all media in parallel
        visual_elements, audio_narration = await asyncio.gather(
            asyncio.gather(*image_tasks),
            asyncio.gather(*audio_tasks)
        )
        
        # Wire imageUrl and audioUrl into each page
        for idx, page in enumerate(pages):
            if idx < len(visual_elements):
                page["imageUrl"] = visual_elements[idx]
            # Find matching audio narration by index
            audio = next((a for a in audio_narration if a.get("page_index") == idx), None)
            if audio:
                page["audioUrl"] = audio["audio_url"]
        
        log_memory_usage("narrative_engine.FableFactory._generate_retry_story_with_media: end")
        return story_data
    
    def _create_readability_feedback_prompt(self, original_prompt: str, target_level: int, current_score: float, previous_text: str) -> str:
        """Create a prompt for readability adjustment based on previous attempt."""
        
        if current_score > target_level:
            adjustment = "SIMPLER"
            direction = "TOO COMPLEX"
            guidance = """
- Use shorter sentences
- Replace complex words with simpler alternatives
- Use fewer syllables per word
- Break up long sentences into multiple shorter ones
- Use more common, everyday vocabulary"""
        else:
            adjustment = "MORE COMPLEX"
            direction = "TOO SIMPLE"  
            guidance = """
- Use slightly longer sentences
- Include some more sophisticated vocabulary
- Add descriptive words and phrases
- Combine some short sentences into longer ones
- Use more varied sentence structures"""
        
        sample_story = get_sample_story(target_level)
        
        target_guidance = self.difficulty_guidance.get(target_level, self.difficulty_guidance[2])
        
        return f"""
You are rewriting a children's story that needs readability adjustment.

READABILITY ANALYSIS OF PREVIOUS ATTEMPT:
- Target Flesch-Kincaid Grade Level: {target_level}
- Actual Score: {current_score:.2f}
- Status: {direction} (difference: {abs(current_score - target_level):.2f} grade levels)

TARGET REQUIREMENTS:
- Word Count: {target_guidance['target_words']} words (for 5-minute reading at {target_guidance['reading_speed']})
- Sentence Structure: {target_guidance['sentence_length']}
- Vocabulary Level: {target_guidance['vocabulary']}

ADJUSTMENT NEEDED: Make the story {adjustment}

SPECIFIC GUIDANCE:{guidance}

TARGET EXAMPLE FOR GRADE LEVEL {target_level}:
Title: "{sample_story['title']}"
Text: "{sample_story['text']}"

PREVIOUS STORY TEXT (for reference):
{previous_text}

TASK: Rewrite the story to better match Grade Level {target_level}. Keep the same basic plot and characters, but adjust the language complexity to target {target_guidance['target_words']} words total. Ensure your rewritten story would score closer to {target_level} on the Flesch-Kincaid scale.

Create detailed character descriptions and image prompts as before. Return the response in the same JSON format with title, characters, and pages.
"""

    async def generate_story_with_translations(self, prompt: str, target_languages: List[SupportedLanguage] = None) -> dict:
        """
        Generate a story package with optional translations to multiple languages.
        
        Args:
            prompt: The story generation prompt
            target_languages: List of languages to translate to (excluding English)
            
        Returns:
            Dictionary containing story data for all requested languages
        """
        # Generate the original English story
        english_story = await self.generate_story_package(prompt)
        
        if not target_languages:
            return {"en": english_story}
        
        # Initialize translation service
        translation_service = TranslationService()
        
        # Create translation tasks for each target language
        translation_tasks = []
        for language in target_languages:
            if language != SupportedLanguage.ENGLISH:
                task = self._generate_translated_story(english_story, language, translation_service)
                translation_tasks.append((language, task))
        
        # Execute all translations in parallel
        if translation_tasks:
            logger.info(f"Starting translation to {len(translation_tasks)} languages")
            translation_results = await asyncio.gather(
                *[task for _, task in translation_tasks], 
                return_exceptions=True
            )
            
            # Compile results
            story_data = {"en": english_story}
            
            for i, (language, _) in enumerate(translation_tasks):
                result = translation_results[i]
                if isinstance(result, Exception):
                    logger.error(f"Translation to {language.value} failed: {result}")
                    # Use English as fallback
                    story_data[language.value] = english_story
                else:
                    story_data[language.value] = result
                    
            return story_data
        
        return {"en": english_story}
    
    async def _generate_translated_story(self, english_story: dict, target_language: SupportedLanguage, translation_service: TranslationService) -> dict:
        """
        Generate a translated version of the story with new audio.
        
        Args:
            english_story: The original English story data
            target_language: Target language for translation
            translation_service: Translation service instance
            
        Returns:
            Translated story dictionary
        """
        # Translate all page texts
        pages = english_story.get("pages", [])
        translated_pages = await translation_service.translate_story_pages(pages, target_language)
        
        # Generate audio for translated pages using Nova voice
        audio_tasks = [
            self._generate_single_narration(page["translated_text"], idx, "nova")
            for idx, page in enumerate(translated_pages)
        ]
        
        logger.info(f"Generating audio for {len(translated_pages)} translated pages in {target_language.value}")
        translated_audio = await asyncio.gather(*audio_tasks, return_exceptions=True)
        
        # Wire up the translated audio URLs
        for idx, page in enumerate(translated_pages):
            # Keep original image (reuse across languages)
            if idx < len(translated_audio) and not isinstance(translated_audio[idx], Exception):
                audio_result = translated_audio[idx]
                page["audioUrl"] = audio_result.get("audio_url")
            else:
                logger.warning(f"Failed to generate audio for page {idx} in {target_language.value}")
                page["audioUrl"] = None
        
        # Create translated story structure
        translated_story = {
            "title": english_story.get("title"),  # Keep English title for now
            "characters": english_story.get("characters"),  # Keep English character descriptions
            "pages": translated_pages,
            "visual_elements": english_story.get("visual_elements"),  # Reuse images
            "audio_narration": [{"page_index": i, "audio_url": page.get("audioUrl")} for i, page in enumerate(translated_pages)],
            "word_count": sum(len(page.get("translated_text", "").split()) for page in translated_pages),
            "illustration_count": english_story.get("illustration_count"),
            "page_count": len(translated_pages),
            "language": target_language.value
        }
        
        return translated_story

    async def _generate_single_illustration(self, page_prompt, idx, art_direction):
        log_memory_usage(f"narrative_engine.FableFactory._generate_single_illustration: start idx={idx}")
        if os.environ.get("USE_DUMMY_AI", "false").lower() in ("true", "1", "yes") or \
           os.environ.get("MOCK_IMAGES", "false").lower() in ("true", "1", "yes"):
            return f"https://kids-story-assets-dev.s3.us-west-1.amazonaws.com/images/dummy-illustration-{idx}.webp"
        prompt = f"Colorful children's book illustration, {art_direction}: {page_prompt}"
        resp = await asyncio.to_thread(self.client.images.generate, model="dall-e-3", prompt=prompt, size="1024x1024", n=1)
        url = resp.data[0].url
        # Download image bytes
        img_resp = await asyncio.to_thread(requests.get, url)
        img_bytes = img_resp.content
        # Convert PNG bytes to WebP in-memory
        with Image.open(io.BytesIO(img_bytes)) as im:
            if im.mode in ("RGBA", "P"):
                im = im.convert("RGB")
            webp_buf = io.BytesIO()
            im.save(webp_buf, format="WEBP", quality=85)
            webp_bytes = webp_buf.getvalue()
        s3_url = upload_file_to_s3(webp_bytes, file_type="images", extension="webp")
        return s3_url

    def _select_voice_for_story(self, text):
        """
        DEPRECATED: Always use Nova voice for consistency across languages.
        This method is kept for backward compatibility but will always return 'nova'.
        """
        return "nova"

    async def _generate_single_narration(self, page_text, idx, voice):
        log_memory_usage(f"narrative_engine.FableFactory._generate_single_narration: start idx={idx}")
        """
        Generate TTS audio for a page using OpenAI TTS and return S3 audio URL and page index (no bytes in result).
        """
        if os.environ.get("USE_DUMMY_AI", "false").lower() in ("true", "1", "yes") or \
           os.environ.get("MOCK_AUDIO", "false").lower() in ("true", "1", "yes"):
            return {"audio_url": f"https://kids-story-assets-dev.s3.us-west-1.amazonaws.com/audio/dummy-narration-{idx}.mp3", "page_index": idx}
        
        client = getattr(self, "client", openai)
        resp = await asyncio.to_thread(
            client.audio.speech.create,
            model="tts-1",
            voice=voice,
            input=page_text
        )
        audio_bytes = resp.content if hasattr(resp, "content") else resp
        audio_url = upload_file_to_s3(audio_bytes, file_type="audio", extension="mp3")
        return {"audio_url": audio_url, "page_index": idx}

def generate_story_package(prompt: str) -> dict:
    """
    Synchronous wrapper for Lambda compatibility.
    Orchestrates story generation using FableFactory.
    """
    factory = FableFactory()
    return asyncio.run(factory.generate_story_package(prompt))

# All logic now lives in FableFactory methods above.
