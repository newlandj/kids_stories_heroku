import asyncio
import logging
from typing import Dict, List, Optional

from openai import AsyncOpenAI

from app.models import SupportedLanguage
from app.settings import AppConfig

logger = logging.getLogger(__name__)

OPEN_AI_MODEL = "gpt-4.1-mini"


class TranslationService:
    """Service for translating story content using OpenAI LLMs."""

    def __init__(self):
        self.openai_key = AppConfig.get_value("openai_key")
        self.client = AsyncOpenAI(api_key=self.openai_key)

        # Language mapping for better prompts
        self.language_names = {
            SupportedLanguage.ENGLISH: "English",
            SupportedLanguage.SPANISH: "Spanish",
            SupportedLanguage.CHINESE: "Chinese (Simplified)",
            SupportedLanguage.FRENCH: "French",
            SupportedLanguage.PORTUGUESE: "Portuguese",
            SupportedLanguage.KOREAN: "Korean",
            SupportedLanguage.HINDI: "Hindi",
        }

    async def translate_page_text(
        self,
        text: str,
        target_language: SupportedLanguage,
        difficulty_level: int = None,
        retries: int = 3,
    ) -> Optional[str]:
        """
        Translate a single page of story text to the target language.

        Args:
            text: The original English text to translate
            target_language: The target language enum
            difficulty_level: Target reading level (0-8) for age-appropriate translation
            retries: Number of retry attempts on failure

        Returns:
            Translated text or None if translation fails
        """
        if target_language == SupportedLanguage.ENGLISH:
            return text  # No translation needed

        language_name = self.language_names[target_language]

        # Add difficulty level guidance if provided
        difficulty_guidance = ""
        if difficulty_level is not None:
            if difficulty_level == 0:
                difficulty_guidance = "Use very simple sentences (3-5 words) and basic vocabulary suitable for ages 3-4."
            elif difficulty_level <= 2:
                difficulty_guidance = "Use short sentences (4-6 words) and simple vocabulary suitable for ages 4-6."
            elif difficulty_level <= 4:
                difficulty_guidance = "Use varied sentence lengths (6-10 words) and grade-appropriate vocabulary suitable for ages 6-8."
            elif difficulty_level <= 6:
                difficulty_guidance = "Use more complex sentences (8-15 words) and richer vocabulary suitable for ages 8-10."
            else:
                difficulty_guidance = "Use sophisticated sentence structures (10-18 words) and advanced vocabulary suitable for ages 10-12."

        # Add language-specific cultural and formatting guidance
        cultural_guidance = ""
        if target_language == SupportedLanguage.KOREAN:
            cultural_guidance = "Use age-appropriate Korean vocabulary and consider Korean cultural context. Use proper Hangul script formatting."
        elif target_language == SupportedLanguage.HINDI:
            cultural_guidance = "Use age-appropriate Hindi vocabulary and consider Indian cultural context. Use proper Devanagari script formatting."

        prompt = f"""Translate the following children's story text to {language_name}. 
Maintain the same tone and reading level as the original text.
{difficulty_guidance}
{cultural_guidance}

Original text:
{text}

Translated text:"""

        for attempt in range(retries):
            try:
                response = await self.client.responses.create(
                    model=OPEN_AI_MODEL,
                    input=[
                        {
                            "role": "system",
                            "content": "You are a professional translator specializing in children's literature. Provide only the translated text without any additional commentary.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.3,  # Lower temperature for more consistent translations
                )

                # Extract the translated text from the response using the new format
                if (
                    response.output
                    and len(response.output) > 0
                    and response.output[0].content
                    and len(response.output[0].content) > 0
                ):
                    translated_text = response.output[0].content[0].text.strip()
                    logger.info(f"Successfully translated text to {language_name}")
                    return translated_text
                else:
                    raise ValueError(
                        "Unexpected response format: missing translation text"
                    )

            except Exception as e:
                logger.warning(f"Translation attempt {attempt + 1} failed: {e}")
                if attempt == retries - 1:
                    logger.error(
                        f"Failed to translate text to {language_name} after {retries} attempts"
                    )
                    return None
                await asyncio.sleep(1)  # Brief delay before retry

        return None

    async def translate_story_pages(
        self,
        pages: List[Dict],
        target_language: SupportedLanguage,
        difficulty_level: int = None,
    ) -> List[Dict]:
        """
        Translate all pages of a story in parallel.

        Args:
            pages: List of page dictionaries with 'text', 'order', etc.
            target_language: Target language enum
            difficulty_level: Target reading level (0-8) for age-appropriate translation

        Returns:
            List of translated page dictionaries
        """
        if target_language == SupportedLanguage.ENGLISH:
            return pages  # No translation needed

        logger.info(
            f"Starting translation of {len(pages)} pages to {self.language_names[target_language]}"
        )

        # Create translation tasks for all pages
        translation_tasks = [
            self._translate_page_with_metadata(page, target_language, difficulty_level)
            for page in pages
        ]

        # Execute translations in parallel
        translated_pages = await asyncio.gather(
            *translation_tasks, return_exceptions=True
        )

        # Filter out failed translations and log results
        successful_translations = []
        successful_count = 0
        failed_count = 0

        for i, result in enumerate(translated_pages):
            if isinstance(result, Exception):
                logger.error(f"Failed to translate page {i}: {result}")
                # Use original English text as fallback
                fallback_page = pages[i].copy()
                fallback_page["translated_text"] = pages[i]["text"]
                fallback_page["translation_failed"] = True
                successful_translations.append(fallback_page)
                failed_count += 1
            else:
                successful_translations.append(result)
                if not result.get("translation_failed", False):
                    successful_count += 1

        logger.info(
            f"Translation completed: {successful_count}/{len(pages)} successful, {failed_count} fallbacks"
        )
        return successful_translations

    async def _translate_page_with_metadata(
        self,
        page: Dict,
        target_language: SupportedLanguage,
        difficulty_level: int = None,
    ) -> Dict:
        """
        Translate a single page and preserve metadata.

        Args:
            page: Page dictionary with text and metadata
            target_language: Target language enum
            difficulty_level: Target reading level (0-8) for age-appropriate translation

        Returns:
            Page dictionary with translated text
        """
        translated_text = await self.translate_page_text(
            page["text"], target_language, difficulty_level
        )

        # Create new page dict with translation
        translated_page = page.copy()
        translated_page["translated_text"] = (
            translated_text or page["text"]
        )  # Fallback to original
        translated_page["target_language"] = target_language.value
        translated_page["translation_failed"] = translated_text is None

        return translated_page

    def get_supported_languages(self) -> List[SupportedLanguage]:
        """Get list of supported languages for translation."""
        return [lang for lang in SupportedLanguage if lang != SupportedLanguage.ENGLISH]
