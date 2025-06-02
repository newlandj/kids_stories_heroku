import asyncio
import logging
from typing import List, Dict, Optional
from openai import AsyncOpenAI
from app.models import SupportedLanguage
from app.settings import AppConfig

logger = logging.getLogger(__name__)

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
            SupportedLanguage.PORTUGUESE: "Portuguese"
        }
    
    async def translate_page_text(self, text: str, target_language: SupportedLanguage, retries: int = 3) -> Optional[str]:
        """
        Translate a single page of story text to the target language.
        
        Args:
            text: The original English text to translate
            target_language: The target language enum
            retries: Number of retry attempts on failure
            
        Returns:
            Translated text or None if translation fails
        """
        if target_language == SupportedLanguage.ENGLISH:
            return text  # No translation needed
            
        language_name = self.language_names[target_language]
        
        prompt = f"""Translate the following children's story text to {language_name}. 
Maintain the same tone, reading level, and child-friendly language. 
Keep the translation appropriate for young children (ages 3-8).

Original text:
{text}

Translated text:"""

        for attempt in range(retries):
            try:
                response = await self.client.chat.completions.create(
                    model="gpt-4o-mini",  # Cost-effective model for translation
                    messages=[
                        {"role": "system", "content": "You are a professional translator specializing in children's literature. Provide only the translated text without any additional commentary."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=500,
                    temperature=0.3  # Lower temperature for more consistent translations
                )
                
                translated_text = response.choices[0].message.content.strip()
                logger.info(f"Successfully translated text to {language_name}")
                return translated_text
                
            except Exception as e:
                logger.warning(f"Translation attempt {attempt + 1} failed: {e}")
                if attempt == retries - 1:
                    logger.error(f"Failed to translate text to {language_name} after {retries} attempts")
                    return None
                await asyncio.sleep(1)  # Brief delay before retry
        
        return None
    
    async def translate_story_pages(self, pages: List[Dict], target_language: SupportedLanguage) -> List[Dict]:
        """
        Translate all pages of a story in parallel.
        
        Args:
            pages: List of page dictionaries with 'text', 'order', etc.
            target_language: Target language enum
            
        Returns:
            List of translated page dictionaries
        """
        if target_language == SupportedLanguage.ENGLISH:
            return pages  # No translation needed
            
        logger.info(f"Starting translation of {len(pages)} pages to {self.language_names[target_language]}")
        
        # Create translation tasks for all pages
        translation_tasks = [
            self._translate_page_with_metadata(page, target_language)
            for page in pages
        ]
        
        # Execute translations in parallel
        translated_pages = await asyncio.gather(*translation_tasks, return_exceptions=True)
        
        # Filter out failed translations and log results
        successful_translations = []
        failed_count = 0
        
        for i, result in enumerate(translated_pages):
            if isinstance(result, Exception):
                logger.error(f"Failed to translate page {i}: {result}")
                # Use original English text as fallback
                fallback_page = pages[i].copy()
                fallback_page['translated_text'] = pages[i]['text']
                fallback_page['translation_failed'] = True
                successful_translations.append(fallback_page)
                failed_count += 1
            else:
                successful_translations.append(result)
        
        logger.info(f"Translation completed: {len(successful_translations) - failed_count}/{len(pages)} successful, {failed_count} fallbacks")
        return successful_translations
    
    async def _translate_page_with_metadata(self, page: Dict, target_language: SupportedLanguage) -> Dict:
        """
        Translate a single page and preserve metadata.
        
        Args:
            page: Page dictionary with text and metadata
            target_language: Target language enum
            
        Returns:
            Page dictionary with translated text
        """
        translated_text = await self.translate_page_text(page['text'], target_language)
        
        # Create new page dict with translation
        translated_page = page.copy()
        translated_page['translated_text'] = translated_text or page['text']  # Fallback to original
        translated_page['target_language'] = target_language.value
        translated_page['translation_failed'] = translated_text is None
        
        return translated_page
    
    def get_supported_languages(self) -> List[SupportedLanguage]:
        """Get list of supported languages for translation."""
        return [lang for lang in SupportedLanguage if lang != SupportedLanguage.ENGLISH]
