"""
AI pipeline logic:
- USE_DUMMY_AI=true: Use dummy images/audio (for fast local/dev testing, no API calls)
"""

import logging
import time

# Configure logging early
logger = logging.getLogger("kids-story-app")


import asyncio
import base64
import os
from typing import List, Optional

from openai import OpenAI
from pydantic import BaseModel

from app.content_safety import ContentScreener
from app.models import SupportedLanguage
from app.model_providers import (
    ModelProviderFactory, 
    ModelPreferences, 
    ModelConfig,
    TextModel,
    ImageModel, 
    AudioModel
)
from app.readability_analyzer import ReadabilityAnalyzer
from app.sample_stories import get_sample_story
from app.storage import upload_file_to_s3
from app.translation_service import TranslationService
from app.utils import log_memory_usage

# Deprecated - keeping for backward compatibility
OPEN_AI_MODEL = "gpt-4o"  # Alternates: gpt-4o, gpt-4o-mini, gpt-4.1-mini


# Pydantic models for structured outputs
class Character(BaseModel):
    name: str
    description: str


class StoryPage(BaseModel):
    text: str
    imagePrompt: str


class StoryStructure(BaseModel):
    title: str
    characters: List[Character]
    pages: List[StoryPage]


class FableFactory:
    def __init__(self, model_preferences: Optional[ModelPreferences] = None):
        # Check if using dummy AI first
        self.use_dummy_ai = os.environ.get("USE_DUMMY_AI", "false").lower() in (
            "true",
            "1",
            "yes",
        )

        # Store model preferences or use defaults
        self.model_preferences = model_preferences or ModelPreferences()
        
        # Initialize legacy OpenAI client for backward compatibility
        if not self.use_dummy_ai:
            api_key = self._get_openai_api_key()

            if not api_key:
                logger.error("No OpenAI API key found")
                raise ValueError("OpenAI API key is required")

            self.client = OpenAI(api_key=api_key, timeout=60.0)
        else:
            self.client = None

        self.content_screener = ContentScreener()
        self.readability_analyzer = ReadabilityAnalyzer()
        # Define difficulty guidance for all 9 levels (ages 3-12)
        self.difficulty_guidance = {
            0: {  # Age 3-4
                "age": "3-4",
                "description": "Toddler/Preschool",
                "sentence_length": "Very short sentences (3-5 words each)",
                "vocabulary": "Simple, familiar words only (cat, dog, mom, run, play)",
                "concepts": "Basic concepts like colors, animals, family",
                "target_words": 150,
                "reading_speed": 40,  # words per minute when read to
                "max_pages": 4,
            },
            1: {  # Age 4-5
                "age": "4-5",
                "description": "Pre-K/Kindergarten",
                "sentence_length": "Short sentences (4-6 words each)",
                "vocabulary": "Simple vocabulary with some repetition",
                "concepts": "Everyday activities, simple emotions",
                "target_words": 200,
                "reading_speed": 50,
                "max_pages": 4,
            },
            2: {  # Age 5-6
                "age": "5-6",
                "description": "Kindergarten/Early reading",
                "sentence_length": "Simple sentences (5-8 words each)",
                "vocabulary": "Basic sight words and phonetic patterns",
                "concepts": "Simple problem-solving, friendship themes",
                "target_words": 250,
                "reading_speed": 60,
                "max_pages": 4,
            },
            3: {  # Age 6-7
                "age": "6-7",
                "description": "First grade/Beginning reader",
                "sentence_length": "Mix of short and medium sentences (6-10 words)",
                "vocabulary": "Grade-appropriate vocabulary with some challenging words",
                "concepts": "Simple adventures, basic moral lessons",
                "target_words": 350,
                "reading_speed": 80,
                "max_pages": 4,
            },
            4: {  # Age 7-8
                "age": "7-8",
                "description": "Second grade/Developing reader",
                "sentence_length": "Varied sentence lengths (7-12 words)",
                "vocabulary": "More diverse vocabulary, compound words",
                "concepts": "Character development, cause and effect",
                "target_words": 450,
                "reading_speed": 100,
                "max_pages": 5,
            },
            5: {  # Age 9
                "age": "9",
                "description": "Advanced reading",
                "sentence_length": "Complex sentences with varied structures (8-15 words)",
                "vocabulary": "Rich vocabulary, descriptive language",
                "concepts": "Multiple characters, subplots, deeper themes",
                "target_words": 600,
                "reading_speed": 130,
                "max_pages": 6,
            },
            6: {  # Age 10
                "age": "10",
                "description": "Complex stories",
                "sentence_length": "Sophisticated sentence structures (10-18 words)",
                "vocabulary": "Advanced vocabulary, figurative language",
                "concepts": "Complex relationships, moral dilemmas",
                "target_words": 750,
                "reading_speed": 155,
                "max_pages": 7,
            },
            7: {  # Age 11
                "age": "11",
                "description": "Pre-teen literature",
                "sentence_length": "Advanced sentence complexity (12-20 words)",
                "vocabulary": "Mature vocabulary, abstract concepts",
                "concepts": "Character growth, complex themes, multiple perspectives",
                "target_words": 875,
                "reading_speed": 175,
                "max_pages": 7,
            },
            8: {  # Age 12
                "age": "12",
                "description": "Middle grade literature",
                "sentence_length": "Adult-like sentence structures (15-25 words)",
                "vocabulary": "Sophisticated vocabulary, nuanced expressions",
                "concepts": "Deep themes, complex character development, mature subjects",
                "target_words": 975,
                "reading_speed": 195,
                "max_pages": 7,
            },
        }

    async def generate_story_package(
        self, prompt: str, difficulty_level: int = None, model_preferences: Optional[ModelPreferences] = None
    ) -> dict:
        log_memory_usage("narrative_engine.FableFactory.generate_story_package: start")
        start_time = time.monotonic()
        
        # Update model preferences if provided
        if model_preferences:
            self.model_preferences = model_preferences

        # Step 1: Generate the basic story structure first
        story = await self.weave_narrative(prompt, difficulty_level)
        if not story or not story.get("pages"):
            raise ValueError("Failed to generate story structure")

        pages = story.get("pages", [])
        log_memory_usage(
            "narrative_engine.FableFactory.generate_story_package: after weave_narrative"
        )

        # Prepare character descriptions for consistent imagery
        character_descriptions = ", ".join(
            [f"{c['name']}: {c['description']}" for c in story.get("characters", [])]
        )

        # Step 2: Generate images in small batches to balance speed and rate limits
        visual_elements = []
        batch_size = 2  # Generate 2 images at a time

        for i in range(0, len(pages), batch_size):
            batch_pages = pages[i : i + batch_size]
            batch_tasks = []

            for j, page in enumerate(batch_pages):
                idx = i + j
                task = self._generate_single_illustration_with_retry(
                    f"{character_descriptions}. {page['imagePrompt']}",
                    idx,
                    art_direction=None,
                )
                batch_tasks.append((idx, task))

            # Execute batch in parallel
            batch_results = await asyncio.gather(
                *[task for _, task in batch_tasks], return_exceptions=True
            )

            # Process results
            for j, ((idx, _), result) in enumerate(
                zip(batch_tasks, batch_results, strict=False)
            ):
                if isinstance(result, Exception):
                    logger.error(f"Failed to generate image for page {idx}: {result}")
                    visual_elements.append(
                        f"https://kids-story-assets-dev.s3.us-west-1.amazonaws.com/images/placeholder-{idx}.webp"
                    )
                else:
                    visual_elements.append(result)

            # Small delay between batches (but not after the last batch)
            if i + batch_size < len(pages):
                await asyncio.sleep(1)

        # Step 3: Generate audio in parallel (audio API is more forgiving)
        voice = self._select_voice_for_story(pages[0]["text"] if pages else "")
        audio_tasks = [
            self._generate_single_narration(page["text"], idx, voice)
            for idx, page in enumerate(pages)
        ]
        audio_narration = await asyncio.gather(*audio_tasks, return_exceptions=True)

        log_memory_usage(
            "narrative_engine.FableFactory.generate_story_package: after images/audio"
        )

        # Step 4: Wire URLs into pages
        for idx, page in enumerate(pages):
            if idx < len(visual_elements):
                page["imageUrl"] = visual_elements[idx]
            # Handle audio results (could be exceptions)
            if idx < len(audio_narration) and not isinstance(
                audio_narration[idx], Exception
            ):
                audio = audio_narration[idx]
                page["audioUrl"] = audio["audio_url"]
            else:
                logger.warning(f"Failed to generate audio for page {idx}")
                page["audioUrl"] = None

        log_memory_usage(
            "narrative_engine.FableFactory.generate_story_package: after wiring URLs"
        )

        word_count = sum(len(page["text"].split()) for page in pages)
        illustration_count = len(visual_elements)
        page_count = len(pages)
        elapsed = time.monotonic() - start_time

        logger.info(
            f"Story generation complete in {elapsed:.2f} seconds. Title: {story.get('title')}, {page_count} pages, {illustration_count} images, {word_count} words."
        )
        log_memory_usage("narrative_engine.FableFactory.generate_story_package: end")

        # Add model tracking metadata
        model_metadata = {
            "text_model": self.model_preferences.text_model.value,
            "image_model": self.model_preferences.image_model.value,
            "audio_model": self.model_preferences.audio_model.value,
            "creation_duration": elapsed,
            "correct_first_try": True  # Will be updated by readability checking logic
        }

        return {
            "title": story.get("title"),
            "characters": story.get("characters"),
            "pages": pages,
            "visual_elements": visual_elements,
            "audio_narration": [
                a for a in audio_narration if not isinstance(a, Exception)
            ],
            "word_count": word_count,
            "illustration_count": illustration_count,
            "page_count": page_count,
            "model_metadata": model_metadata,
        }

    """Orchestrates the generation of children's story elements using AI"""

    def _get_openai_api_key(self) -> Optional[str]:
        return os.environ.get("OPENAI_API_KEY")

    def _get_dummy_story_data(self) -> dict:
        """Return dummy story data for testing/development."""
        return {
            "title": "The Brave Child's Adventure",
            "characters": [
                {
                    "name": "Jamie",
                    "description": "a brave 8-year-old with curly brown hair and a blue backpack",
                }
            ],
            "pages": [
                {
                    "text": "Jamie found a mysterious map in the attic.",
                    "imagePrompt": "A child holding a map in a dusty attic, sunlight streaming in.",
                },
                {
                    "text": "Jamie followed the map into the woods, meeting a talking squirrel.",
                    "imagePrompt": "A child and a talking squirrel in a magical forest.",
                },
                {
                    "text": "Together, they discovered a hidden treasure.",
                    "imagePrompt": "A child and squirrel celebrating next to a treasure chest in the woods.",
                },
            ],
        }

    def _get_dummy_story_with_media(self) -> dict:
        """Return dummy story data with media URLs for testing/development."""
        return {
            "title": "The Brave Child's Adventure (Retry)",
            "characters": [
                {
                    "name": "Jamie",
                    "description": "a brave 8-year-old with curly brown hair and a blue backpack",
                }
            ],
            "pages": [
                {
                    "text": "Jamie found a map.",
                    "imagePrompt": "A child holding a map.",
                    "imageUrl": "https://dummy-image-1.png",
                    "audioUrl": "https://dummy-audio-1.mp3",
                },
                {
                    "text": "Jamie went to the woods.",
                    "imagePrompt": "A child walking into a forest.",
                    "imageUrl": "https://dummy-image-2.png",
                    "audioUrl": "https://dummy-audio-2.mp3",
                },
                {
                    "text": "Jamie found treasure.",
                    "imagePrompt": "A child celebrating next to a treasure chest.",
                    "imageUrl": "https://dummy-image-3.png",
                    "audioUrl": "https://dummy-audio-3.mp3",
                },
            ],
        }

    def _build_story_generation_prompt(self, difficulty_level: int = None) -> str:
        """Build the system prompt for story generation based on difficulty level."""

        if (
            difficulty_level is not None
            and difficulty_level in self.difficulty_guidance
        ):
            level_info = self.difficulty_guidance[difficulty_level]
            target_words = level_info.get("target_words", 600)
            max_pages = level_info.get("max_pages", 4)

            base_prompt = (
                "You are a children's story writer creating engaging, age-appropriate stories for children. "
                f"Create stories with EXACTLY {max_pages} pages (no more, no less), each with a vivid scene that can be illustrated. "
                "For each story, create detailed character descriptions that should remain consistent throughout the story. "
                "For each page, provide a detailed image description that maintains character consistency. "
                "IMPORTANT: Each story MUST include a creative, appropriate book title."
            )

            difficulty_prompt = f"""
            STRICT PAGE LIMIT: Your story must have EXACTLY {max_pages} pages. Do not exceed this limit.
            Target reading level: Grade {difficulty_level} (Age {level_info["age"]})
            Description: {level_info["description"]}
            Sentence length: {level_info["sentence_length"]}
            Vocabulary: {level_info["vocabulary"]}
            Concepts: {level_info["concepts"]}
            Target total word count: {target_words} words across all {max_pages} pages
            """
            result = f"{base_prompt}\n\n{difficulty_prompt}"
        else:
            logger.info("_build_story_generation_prompt: Using default fallback prompt")
            # Default fallback for when no difficulty level is specified
            result = (
                "You are a children's story writer creating engaging, age-appropriate stories for children. "
                "Create stories with EXACTLY 4 pages (no more, no less), each with a vivid scene that can be illustrated. "
                "For each story, create detailed character descriptions that should remain consistent throughout the story. "
                "For each page, provide a detailed image description that maintains character consistency. "
                "IMPORTANT: Each story MUST include a creative, appropriate book title."
            )

        return result

    async def weave_narrative(self, prompt: str, difficulty_level: int = None) -> dict:
        log_memory_usage("narrative_engine.FableFactory.weave_narrative: start")
        if self.use_dummy_ai:
            # Minimal dummy response in new structure
            return self._get_dummy_story_data()
        self.content_screener.validate_prompt(prompt)

        # Build system prompt with difficulty level guidance
        system_prompt = self._build_story_generation_prompt(difficulty_level)
        user_message = f"Create a children's story about: {prompt}. Please ensure your response includes a creative, appropriate title for the book."

        # Get text provider for the selected model
        text_provider = ModelProviderFactory.get_text_provider(self.model_preferences.text_model)
        
        attempts, backoff = 3, 1
        for i in range(attempts):
            try:
                # Use provider's generate_text method with structured output
                response = await text_provider.generate_text(
                    prompt=f"{system_prompt}\n\n{user_message}",
                    model=self.model_preferences.text_model.value,
                    response_schema=StoryStructure,
                    temperature=0.7 + i * 0.1
                )

                # Extract structured output
                if "structured_output" in response:
                    story_data = response["structured_output"]
                    
                    # Convert to dict format for compatibility with existing code
                    return {
                        "title": story_data.title,
                        "characters": [
                            {"name": char.name, "description": char.description}
                            for char in story_data.characters
                        ],
                        "pages": [
                            {"text": page.text, "imagePrompt": page.imagePrompt}
                            for page in story_data.pages
                        ],
                    }
                else:
                    # Fallback to raw content parsing if structured output not available
                    logger.warning("No structured output available, attempting to parse raw content")
                    raise ValueError("Structured output not available")

            except Exception as e:
                logger.error(f"weave_narrative error {i} with {self.model_preferences.text_model}: {e}")
                if i < attempts - 1:
                    await asyncio.sleep(backoff)
                    backoff *= 2

        # Raise exception if all attempts fail
        logger.error("All structured output attempts failed")
        raise Exception("Failed to generate story after all retry attempts")

    def _get_difficulty_guidance(self, difficulty_level: int) -> str:
        """Generate difficulty-specific guidance for the LLM based on Flesch-Kincaid level."""

        level_info = self.difficulty_guidance.get(
            difficulty_level, self.difficulty_guidance[2]
        )  # Default to level 2
        sample_story = get_sample_story(difficulty_level)

        return f"""
IMPORTANT READING LEVEL REQUIREMENTS:
You must write for Flesch-Kincaid Grade Level {difficulty_level} (Age {level_info["age"]} - {level_info["description"]}).

TARGET STORY LENGTH: {level_info["target_words"]} words (for 5-minute reading at {level_info["reading_speed"]})

SPECIFIC WRITING GUIDELINES:
- {level_info["sentence_length"]}
- {level_info["vocabulary"]}
- {level_info["concepts"]}

TARGET FLESCH-KINCAID GRADE LEVEL: {difficulty_level}

EXAMPLE OF APPROPRIATE COMPLEXITY FOR LEVEL {difficulty_level}:
Title: "{sample_story["title"]}"
Text: "{sample_story["text"]}"

Your story should match this level of complexity. Use the Flesch-Kincaid Grade Level formula which considers:
1. Average sentence length (total words ÷ total sentences)
2. Average syllables per word (total syllables ÷ total words)

Keep sentences and vocabulary appropriate for the target age group. Make sure your story would score close to Grade Level {difficulty_level} on the Flesch-Kincaid scale.
"""

    async def generate_story_with_readability_feedback(
        self, prompt: str, target_level: int, current_score: float, previous_story: dict
    ) -> dict:
        """Generate a new story attempt with readability feedback from the previous attempt."""
        log_memory_usage(
            "narrative_engine.FableFactory.generate_story_with_readability_feedback: start"
        )

        if self.use_dummy_ai:
            # Return dummy response with image and audio URLs for testing
            return self._get_dummy_story_with_media()

        self.content_screener.validate_prompt(prompt)

        # Combine all previous story text for analysis
        previous_text = " ".join(
            [page.get("text", "") for page in previous_story.get("pages", [])]
        )

        # Create feedback-based prompt
        feedback_prompt = self._create_readability_feedback_prompt(
            prompt, target_level, current_score, previous_text
        )

        # Generate retry story with structured output using provider system
        attempts, backoff = 3, 1
        for i in range(attempts):
            try:
                # Use the text provider instead of direct OpenAI client
                text_provider = ModelProviderFactory.get_text_provider(self.model_preferences.text_model)
                
                completion = await text_provider.generate_text(
                    prompt=f"{feedback_prompt}\n\nUser: Rewrite the story about: {prompt}. Adjust the complexity to better match Grade Level {target_level}.",
                    model=self.model_preferences.text_model.value,
                    response_schema=StoryStructure,
                    temperature=0.7 + i * 0.1
                )

                # Extract structured output from provider response
                if "structured_output" in completion:
                    story_data = completion["structured_output"]  # Already a StoryStructure object
                else:
                    raise ValueError("No structured output available from provider")

                # Convert to dict format for compatibility with existing code
                return {
                    "title": story_data.title,
                    "characters": [
                        {"name": char.name, "description": char.description}
                        for char in story_data.characters
                    ],
                    "pages": [
                        {"text": page.text, "imagePrompt": page.imagePrompt}
                        for page in story_data.pages
                    ],
                }

            except Exception as e:
                logger.error(f"generate_story_with_readability_feedback error {i}: {e}")
                if i < attempts - 1:
                    await asyncio.sleep(backoff)
                    backoff *= 2

        # If we get here, all attempts failed
        logger.warning("Readability feedback retry failed, returning None")
        return None

    def _create_readability_feedback_prompt(
        self,
        original_prompt: str,
        target_level: int,
        current_score: float,
        previous_text: str,
    ) -> str:
        """Create a prompt for readability adjustment based on previous attempt."""
        start_time = time.monotonic()
        logger.info(
            f"_create_readability_feedback_prompt: Starting with target_level={target_level}, current_score={current_score}"
        )

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

        target_guidance = self.difficulty_guidance.get(
            target_level, self.difficulty_guidance[2]
        )
        max_pages = target_guidance.get("max_pages", 4)

        logger.info(
            f"_create_readability_feedback_prompt: Building feedback prompt, previous_text length: {len(previous_text)} chars"
        )
        result = f"""
You are rewriting a children's story that needs readability adjustment.

READABILITY ANALYSIS OF PREVIOUS ATTEMPT:
- Target Flesch-Kincaid Grade Level: {target_level}
- Actual Score: {current_score:.2f}
- Status: {direction} (difference: {abs(current_score - target_level):.2f} grade levels)

TARGET REQUIREMENTS:
- Page Limit: EXACTLY {max_pages} pages (no more, no less)
- Word Count: {target_guidance["target_words"]} words (for 5-minute reading at {target_guidance["reading_speed"]})
- Sentence Structure: {target_guidance["sentence_length"]}
- Vocabulary Level: {target_guidance["vocabulary"]}

ADJUSTMENT NEEDED: Make the story {adjustment}

SPECIFIC GUIDANCE:{guidance}

TARGET EXAMPLE FOR GRADE LEVEL {target_level}:
Title: "{sample_story["title"]}"
Text: "{sample_story["text"]}"

PREVIOUS STORY TEXT (for reference):
{previous_text}

TASK: Rewrite the story to better match Grade Level {target_level}. Keep the same basic plot and characters, but adjust the language complexity to target {target_guidance["target_words"]} words total across EXACTLY {max_pages} pages. Ensure your rewritten story would score closer to {target_level} on the Flesch-Kincaid scale.

Create detailed character descriptions and image prompts as before. Return the response in the same JSON format with title, characters, and pages.
"""

        elapsed = time.monotonic() - start_time
        logger.info(
            f"_create_readability_feedback_prompt: Completed in {elapsed:.3f} seconds, prompt length: {len(result)} chars"
        )
        return result

    async def generate_story_with_translations(
        self,
        prompt: str,
        target_languages: List[SupportedLanguage] = None,
        english_story: dict = None,
    ) -> dict:
        """
        Generate a story package with optional translations to multiple languages.

        Args:
            prompt: The story generation prompt
            target_languages: List of languages to translate to (excluding English)
            english_story: Optional pre-generated English story to avoid regeneration

        Returns:
            Dictionary containing story data for all requested languages
        """
        # Use provided English story or generate new one
        if english_story is None:
            english_story = await self.generate_story_package(prompt)

        if not target_languages:
            return {"en": english_story}

        # Initialize translation service
        translation_service = TranslationService()

        # Create translation tasks for each target language
        translation_tasks = []
        for language in target_languages:
            if language != SupportedLanguage.ENGLISH:
                task = self._generate_translated_story(
                    english_story, language, translation_service
                )
                translation_tasks.append((language, task))

        # Execute all translations in parallel
        if translation_tasks:
            logger.info(f"Starting translation to {len(translation_tasks)} languages")
            translation_results = await asyncio.gather(
                *[task for _, task in translation_tasks], return_exceptions=True
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

    async def _generate_translated_story(
        self,
        english_story: dict,
        target_language: SupportedLanguage,
        translation_service: TranslationService,
    ) -> dict:
        """
        Generate a translated version of the story without audio initially.
        Reuses English images for consistency. Audio will be generated on-demand.
        """
        # Translate all page texts
        pages = english_story.get("pages", [])
        translated_pages = await translation_service.translate_story_pages(
            pages, target_language
        )

        # Wire up the image URLs from English and set audio to None initially
        english_pages = english_story.get("pages", [])
        for idx, page in enumerate(translated_pages):
            # Inherit image URL from corresponding English page
            if idx < len(english_pages):
                page["imageUrl"] = english_pages[idx].get("imageUrl")

            # Audio will be generated on-demand
            page["audioUrl"] = None

        # Create translated story structure
        translated_story = {
            "title": english_story.get("title"),  # Keep English title for now
            "characters": english_story.get(
                "characters"
            ),  # Keep English character descriptions
            "pages": translated_pages,
            "visual_elements": english_story.get(
                "visual_elements"
            ),  # Reuse English images
            "audio_narration": [],  # Empty since audio is on-demand
            "word_count": sum(
                len(page.get("translated_text", "").split())
                for page in translated_pages
            ),
            "illustration_count": english_story.get(
                "illustration_count"
            ),  # Same as English
            "page_count": len(translated_pages),
            "language": target_language.value,
        }

        return translated_story

    async def _generate_single_illustration_with_retry(
        self, page_prompt, idx, art_direction, max_retries=3
    ):
        """Generate single illustration with retry logic for rate limits"""
        log_memory_usage(
            f"narrative_engine.FableFactory._generate_single_illustration_with_retry: start idx={idx}"
        )

        if (
            self.use_dummy_ai
            or os.environ.get("MOCK_IMAGES", "false").lower() == "true"
        ):
            return "https://kids-story-assets-dev.s3.us-west-1.amazonaws.com/images/dummy-illustration-1.webp"

        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    logger.info(
                        f"Retrying image generation for idx={idx}, attempt {attempt + 1}/{max_retries}"
                    )
                return await self._generate_single_illustration(
                    page_prompt, idx, art_direction
                )
            except Exception as e:
                # Check if it's a rate limit error (429 status code)
                if "429" in str(e) or "rate limit" in str(e).lower():
                    wait_time = (
                        2**attempt
                    ) * 5  # Exponential backoff: 5, 10, 20 seconds
                    logger.warning(
                        f"Rate limit hit for image {idx}, attempt {attempt + 1}/{max_retries}. Waiting {wait_time}s..."
                    )
                    if attempt < max_retries - 1:
                        await asyncio.sleep(wait_time)
                    else:
                        logger.error(f"All retry attempts failed for image {idx}: {e}")
                        raise
                else:
                    logger.error(
                        f"Unexpected error generating image {idx}, attempt {attempt + 1}: {e}"
                    )
                    if attempt == max_retries - 1:
                        raise
                    await asyncio.sleep(2)

    async def _generate_single_illustration(self, page_prompt, idx, art_direction):
        # Removed duplicate logging - now handled by retry method
        if (
            self.use_dummy_ai
            or os.environ.get("MOCK_IMAGES", "false").lower() == "true"
        ):
            return f"https://kids-story-assets-dev.s3.us-west-1.amazonaws.com/images/dummy-illustration-{idx}.webp"

        # Use the provider system for image generation
        prompt = (
            f"Colorful children's book illustration with NO TEXT, NO LETTERS, NO WORDS, NO WRITING of any kind, {art_direction}: {page_prompt}. "
            f"Pure visual illustration only, no text elements, signs, or written characters visible anywhere in the image."
        )
        
        # Get image provider for the selected model
        image_provider = ModelProviderFactory.get_image_provider(self.model_preferences.image_model)
        
        # Generate image using provider
        s3_url = await image_provider.generate_image(
            prompt=prompt,
            model=self.model_preferences.image_model.value
        )
        
        log_memory_usage(
            f"narrative_engine.FableFactory._generate_single_illustration: end idx={idx}"
        )
        return s3_url

    async def _upload_image_to_s3(self, image_data: bytes, filename: str) -> str:
        """Upload image data to S3 and return the URL"""
        try:
            # Use the existing upload_file_to_s3 function
            return upload_file_to_s3(image_data, file_type="images", extension="png")
        except Exception as e:
            logger.error(f"Failed to upload image to S3: {e}")
            # Return a placeholder URL on failure
            return "https://kids-story-assets-dev.s3.us-west-1.amazonaws.com/images/placeholder.png"

    def _select_voice_for_story(self, text):
        """
        DEPRECATED: Always use Nova voice for consistency across languages.
        This method is kept for backward compatibility but will always return 'nova'.
        """
        return "nova"

    def _select_voice_for_language(self, language: str) -> str:
        """
        Select appropriate OpenAI TTS voice based on language.
        OpenAI TTS supports multiple languages with the same voices, but some voices work better for certain languages.
        """
        # OpenAI TTS voices: alloy, echo, fable, onyx, nova, shimmer
        # All voices support multiple languages, but some may sound more natural for specific languages
        voice_mapping = {
            "en": "nova",  # English - clear, friendly voice
            "es": "nova",  # Spanish - nova works well for Spanish
            "fr": "shimmer",  # French - shimmer has good French pronunciation
            "pt": "nova",  # Portuguese - nova works well
            "zh": "alloy",  # Chinese - alloy handles Chinese tones well
            "de": "echo",  # German - echo for German pronunciation
            "it": "fable",  # Italian - fable for Italian flow
            "ja": "alloy",  # Japanese - alloy for Japanese pronunciation
            "ko": "alloy",  # Korean - alloy for Korean pronunciation
        }

        return voice_mapping.get(
            language, "nova"
        )  # Default to nova if language not found

    async def _generate_single_narration(self, page_text, idx, voice, max_retries=3):
        """
        Generate TTS audio for a page using the provider system with robust error handling and retries.
        Returns S3 audio URL and page index.
        """
        log_memory_usage(
            f"narrative_engine.FableFactory._generate_single_narration: start idx={idx}"
        )

        if self.use_dummy_ai or os.environ.get("MOCK_AUDIO", "false").lower() in (
            "true",
            "1",
            "yes",
        ):
            return {
                "audio_url": f"https://kids-story-assets-dev.s3.us-west-1.amazonaws.com/audio/dummy-narration-{idx}.mp3",
                "page_index": idx,
            }

        # Get audio provider for the selected model
        audio_provider = ModelProviderFactory.get_audio_provider(self.model_preferences.audio_model)

        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    logger.info(
                        f"Retrying audio generation for page {idx}, attempt {attempt + 1}/{max_retries}"
                    )
                    await asyncio.sleep(2**attempt)  # Exponential backoff

                # Generate audio using provider
                audio_url = await audio_provider.generate_audio(
                    text=page_text,
                    model=self.model_preferences.audio_model.value,
                    voice=voice,
                    speed=0.9 if hasattr(audio_provider, 'provider_name') and audio_provider.provider_name == 'openai' else None
                )
                
                return {"audio_url": audio_url, "page_index": idx}

            except Exception as e:
                error_msg = str(e).lower()
                if "429" in error_msg or "rate limit" in error_msg:
                    wait_time = (2**attempt) * 5  # Longer backoff for rate limits
                    logger.warning(
                        f"Rate limit hit for audio {idx}, attempt {attempt + 1}/{max_retries}. Waiting {wait_time}s..."
                    )
                    if attempt < max_retries - 1:
                        await asyncio.sleep(wait_time)
                    else:
                        logger.error(f"All retry attempts failed for audio {idx}: {e}")
                        return {"audio_url": None, "page_index": idx, "error": str(e)}
                else:
                    logger.error(
                        f"Error generating audio for page {idx} with {self.model_preferences.audio_model}, attempt {attempt + 1}: {e}"
                    )
                    if attempt == max_retries - 1:
                        return {"audio_url": None, "page_index": idx, "error": str(e)}
                    await asyncio.sleep(2)



    async def weave_narrative_text_only(
        self,
        prompt: str,
        difficulty_level: int = None,
        previous_story: dict = None,
        current_score: float = None,
    ) -> dict:
        """Generate story text only using structured outputs, without images or audio.

        Args:
            prompt: The story generation prompt
            difficulty_level: Target reading level (0-8)
            previous_story: Optional previous story for readability feedback
            current_score: Optional readability score of previous story

        Returns:
            Dictionary containing story structure
        """
        start_time = time.monotonic()
        log_memory_usage(
            "narrative_engine.FableFactory.weave_narrative_text_only: start"
        )


        if self.use_dummy_ai:
            logger.info(
                "weave_narrative_text_only: Using dummy AI, returning dummy data"
            )
            return self._get_dummy_story_data()

        self.content_screener.validate_prompt(prompt)

        # Build system prompt with difficulty level guidance
        system_prompt = self._build_story_generation_prompt(difficulty_level)

        # Add readability feedback if provided
        if previous_story and current_score is not None:
            logger.info(
                "weave_narrative_text_only: Creating readability feedback prompt"
            )
            feedback_prompt = self._create_readability_feedback_prompt(
                prompt,
                difficulty_level,
                current_score,
                " ".join(
                    [page.get("text", "") for page in previous_story.get("pages", [])]
                ),
            )
            system_prompt = feedback_prompt
            user_message = f"Rewrite the story about: {prompt}. Adjust the complexity to better match Grade Level {difficulty_level}."
        else:
            user_message = f"Create a children's story about: {prompt}"

        logger.info(
            f"weave_narrative_text_only: System prompt length: {len(system_prompt)} chars, User message length: {len(user_message)} chars"
        )

        # Generate story with structured output
        attempts, backoff = 3, 1
        for i in range(attempts):
            try:
                logger.info(
                    f"weave_narrative_text_only: Starting OpenAI API call attempt {i + 1}/{attempts}"
                )
                api_call_start = time.monotonic()

                # Use the text provider instead of direct OpenAI client
                text_provider = ModelProviderFactory.get_text_provider(self.model_preferences.text_model)
                logger.info(
                    f"weave_narrative_text_only: Using text provider: {text_provider.__class__.__name__} for model: {self.model_preferences.text_model}"
                )

                # Log the parameters being sent  
                logger.info(
                    f"weave_narrative_text_only: API parameters - temperature={0.7 + i * 0.1}"
                )

                completion = await text_provider.generate_text(
                    prompt=f"{system_prompt}\n\nUser: {user_message}",
                    model=self.model_preferences.text_model.value,
                    response_schema=StoryStructure,
                    temperature=0.7 + i * 0.1
                )

                api_call_elapsed = time.monotonic() - api_call_start
                logger.info(
                    f"weave_narrative_text_only: Total API call (including setup) took {api_call_elapsed:.3f} seconds"
                )

                # Extract structured output from provider response
                if "structured_output" in completion:
                    story_data = completion["structured_output"]  # Already a StoryStructure object
                else:
                    # Fallback - try to parse raw content or raise error
                    raise ValueError("No structured output available from provider")

                # Convert to dict format for compatibility with existing code
                result = {
                    "title": story_data.title,
                    "characters": [
                        {"name": char.name, "description": char.description}
                        for char in story_data.characters
                    ],
                    "pages": [
                        {"text": page.text, "imagePrompt": page.imagePrompt}
                        for page in story_data.pages
                    ],
                }


                total_elapsed = time.monotonic() - start_time
                logger.info(
                    f"weave_narrative_text_only: SUCCESS in {total_elapsed:.3f} seconds. Title: '{result['title']}', {len(result['pages'])} pages"
                )
                return result

            except Exception as e:
                api_call_elapsed = (
                    time.monotonic() - api_call_start
                    if "api_call_start" in locals()
                    else 0
                )
                logger.error(
                    f"weave_narrative_text_only error attempt {i + 1}: {e} (API call took {api_call_elapsed:.3f}s)"
                )
                if i < attempts - 1:
                    logger.info(
                        f"weave_narrative_text_only: Waiting {backoff} seconds before retry"
                    )
                    await asyncio.sleep(backoff)
                    backoff *= 2

        # Raise exception if all attempts fail
        total_elapsed = time.monotonic() - start_time
        logger.error(
            f"weave_narrative_text_only: All structured output attempts failed after {total_elapsed:.3f} seconds"
        )
        raise Exception("Failed to generate story text after all retry attempts")

    async def generate_media_for_story(self, story_data: dict) -> dict:
        """Generate images for a story structure that has only text.
        Audio will be generated on-demand when requested.
        This is for Phase 2 of the optimized workflow - adding media to final chosen text."""
        log_memory_usage(
            "narrative_engine.FableFactory.generate_media_for_story: start"
        )

        pages = story_data.get("pages", [])
        if not pages:
            return story_data

        # Prepare character descriptions for consistent imagery
        character_descriptions = ", ".join(
            [
                f"{c['name']}: {c['description']}"
                for c in story_data.get("characters", [])
            ]
        )

        # Generate only images - audio will be on-demand
        logger.info(
            f"generate_media_for_story: Generating images for {len(pages)} pages"
        )
        image_tasks = [
            self._generate_single_illustration_with_retry(
                f"{character_descriptions}. {page['imagePrompt']}",
                idx,
                art_direction="child-friendly, vibrant colors",
            )
            for idx, page in enumerate(pages)
        ]

        # Execute image generation
        visual_elements = await asyncio.gather(*image_tasks, return_exceptions=True)

        # Process image results and handle any exceptions
        for idx, result in enumerate(visual_elements):
            if isinstance(result, Exception):
                logger.error(f"Failed to generate image for page {idx}: {result}")
                visual_elements[idx] = (
                    f"https://kids-story-assets-dev.s3.us-west-1.amazonaws.com/images/placeholder-{idx}.webp"
                )

        # Wire image URLs into pages (no audio URLs yet)
        for idx, page in enumerate(pages):
            if idx < len(visual_elements):
                page["imageUrl"] = visual_elements[idx]
            # Audio will be generated on-demand, so set to None initially
            page["audioUrl"] = None

        # Return complete story with images only
        result = {
            "title": story_data.get("title"),
            "characters": story_data.get("characters"),
            "pages": pages,
            "visual_elements": visual_elements,
            "audio_narration": [],  # Empty since audio is on-demand
            "word_count": sum(len(page["text"].split()) for page in pages),
            "illustration_count": len(visual_elements),
            "page_count": len(pages),
        }

        log_memory_usage("narrative_engine.FableFactory.generate_media_for_story: end")
        return result

    async def generate_story_with_readability_check_first(
        self,
        prompt: str,
        difficulty_level: int = None,
        target_languages: List[SupportedLanguage] = None,
    ) -> dict:
        """Optimized workflow:
        Phase 1: Generate text only, check readability, retry text if needed
        Phase 2: Generate media AND translations in parallel
        Phase 3: Combine results

        Args:
            prompt: Story generation prompt
            difficulty_level: Target reading level (0-8)
            target_languages: Optional list of languages to translate to
        """
        log_memory_usage(
            "narrative_engine.FableFactory.generate_story_with_readability_check_first: start"
        )
        start_time = time.monotonic()

        # Phase 1: Text Generation with Readability Check
        logger.info("Phase 1: Generating story text and checking readability...")
        phase1_start = time.monotonic()

        # Generate initial story text
        primary_story = await self.weave_narrative_text_only(prompt, difficulty_level)
        if not primary_story or not primary_story.get("pages"):
            raise ValueError("Failed to generate primary story text")

        # Check readability of primary story
        primary_text = " ".join([page["text"] for page in primary_story["pages"]])
        primary_analysis = self.readability_analyzer.analyze_text(primary_text)
        primary_score = (
            primary_analysis.get("grade_level", 2.0) if primary_analysis else 2.0
        )

        target_level = difficulty_level if difficulty_level is not None else 2
        score_difference = abs(primary_score - target_level)

        chosen_story = primary_story

        # If readability is off by more than 1 grade level, generate a retry
        if score_difference > 1.0:
            logger.info(
                f"Primary story readability score {primary_score:.1f} is off target {target_level} by {score_difference:.1f}. Generating retry..."
            )

            try:
                retry_story = await self.weave_narrative_text_only(
                    prompt, target_level, primary_story, primary_score
                )

                if retry_story:
                    retry_text = " ".join(
                        [page["text"] for page in retry_story["pages"]]
                    )
                    retry_analysis = self.readability_analyzer.analyze_text(retry_text)
                    retry_score = (
                        retry_analysis.get("grade_level", 2.0)
                        if retry_analysis
                        else 2.0
                    )
                    retry_difference = abs(retry_score - target_level)

                    # Choose the story with better readability score
                    if retry_difference < score_difference:
                        logger.info(
                            f"Using retry story (score {retry_score:.1f}, diff {retry_difference:.1f}) over primary (score {primary_score:.1f}, diff {score_difference:.1f})"
                        )
                        chosen_story = retry_story
                    else:
                        logger.info(
                            f"Keeping primary story (score {primary_score:.1f}, diff {score_difference:.1f}) over retry (score {retry_score:.1f}, diff {retry_difference:.1f})"
                        )
                else:
                    logger.warning(
                        "Retry story generation failed, keeping primary story"
                    )

            except Exception as e:
                logger.error(f"Error in readability retry: {e}, keeping primary story")
        else:
            logger.info(
                f"Primary story readability score {primary_score:.1f} is close enough to target {target_level} (diff {score_difference:.1f})"
            )

        phase1_elapsed = time.monotonic() - phase1_start
        logger.info(
            f"Phase 1 complete in {phase1_elapsed:.2f} seconds. Chosen story: '{chosen_story.get('title')}'"
        )

        # Phase 2: Generate media AND translations in parallel
        logger.info(
            "Phase 2: Generating images, audio, and translations in parallel..."
        )
        phase2_start = time.monotonic()

        # Start all tasks in parallel:
        # 1. Generate English media (images + audio)
        # 2. Generate translations if needed
        tasks = []

        # Task 1: Generate English media
        media_task = self.generate_media_for_story(chosen_story)
        tasks.append(("en", media_task))

        # Task 2: Generate translations if needed
        if target_languages:
            translation_service = TranslationService()
            for language in target_languages:
                if language != SupportedLanguage.ENGLISH:
                    task = self._generate_translated_story(
                        chosen_story, language, translation_service
                    )
                    tasks.append((language.value, task))

        # Execute all tasks in parallel
        results = await asyncio.gather(
            *[task for _, task in tasks], return_exceptions=True
        )

        # Process results
        story_data = {}
        for (lang, _), result in zip(tasks, results, strict=False):
            if isinstance(result, Exception):
                logger.error(f"Failed to generate {lang} story: {result}")
                if lang == "en":
                    raise ValueError("Failed to generate English story with media")
                # For translations, use English as fallback
                story_data[lang] = chosen_story
            else:
                story_data[lang] = result

        phase2_elapsed = time.monotonic() - phase2_start
        logger.info(f"Phase 2 complete in {phase2_elapsed:.2f} seconds")

        # Summary
        total_elapsed = time.monotonic() - start_time
        word_count = story_data["en"].get("word_count", 0)
        page_count = story_data["en"].get("page_count", 0)
        illustration_count = story_data["en"].get("illustration_count", 0)
        translation_count = len(story_data) - 1  # Subtract English

        logger.info(
            f"Optimized story generation complete in {total_elapsed:.2f} seconds. "
            f"Title: {story_data['en'].get('title')}, {page_count} pages, "
            f"{illustration_count} images, {word_count} words, "
            f"{translation_count} translations."
        )

        log_memory_usage(
            "narrative_engine.FableFactory.generate_story_with_readability_check_first: end"
        )
        return story_data


