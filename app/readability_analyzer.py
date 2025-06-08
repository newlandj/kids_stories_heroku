"""
Readability analysis using Flesch-Kincaid Grade Level formula.
Provides accurate syllable counting and text analysis for story difficulty assessment.
"""

import re
import logging
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


class ReadabilityAnalyzer:
    """
    Analyzes text readability using the Flesch-Kincaid Grade Level formula.
    
    Formula: Grade Level = 0.39 × (words/sentences) + 11.8 × (syllables/words) - 15.59
    """
    
    def __init__(self):
        """Initialize the analyzer with pronunciation dictionary."""
        self.pronunciation_dict = None
        self._initialize_nltk()
    
    def _initialize_nltk(self):
        """Initialize NLTK resources with proper error handling."""
        try:
            import nltk
            from nltk.corpus import cmudict
            
            # Try to load the pronunciation dictionary
            try:
                self.pronunciation_dict = cmudict.dict()
                logger.info("CMU pronunciation dictionary loaded successfully")
            except LookupError:
                logger.info("Downloading CMU pronunciation dictionary...")
                try:
                    # Set NLTK data path to avoid permission issues on Heroku
                    import os
                    nltk_data_path = os.path.join(os.getcwd(), 'nltk_data')
                    if not os.path.exists(nltk_data_path):
                        os.makedirs(nltk_data_path)
                    nltk.data.path.append(nltk_data_path)
                    
                    # Download with timeout and error handling
                    nltk.download('cmudict', quiet=True, download_dir=nltk_data_path)
                    self.pronunciation_dict = cmudict.dict()
                    logger.info("CMU pronunciation dictionary downloaded and loaded")
                except Exception as download_error:
                    logger.warning(f"Failed to download NLTK data: {download_error}")
                    logger.warning("Falling back to simple syllable counting method")
                    self.pronunciation_dict = None
                
        except ImportError:
            logger.error("NLTK not available. Install with: pip install nltk")
            self.pronunciation_dict = None
        except Exception as e:
            logger.error(f"Failed to initialize NLTK: {e}")
            self.pronunciation_dict = None
    
    def count_syllables(self, word: str) -> int:
        """
        Count syllables in a word using CMU pronunciation dictionary with fallback.
        
        Args:
            word: The word to analyze
            
        Returns:
            Number of syllables in the word (minimum 1)
        """
        if not word:
            return 1
            
        # Clean the word
        clean_word = word.lower().strip(".,!?;:\"'()[]{}").strip()
        
        if not clean_word:
            return 1
        
        # Try CMU pronunciation dictionary first
        if self.pronunciation_dict and clean_word in self.pronunciation_dict:
            # Count stress markers (digits) in pronunciation
            pronunciation = self.pronunciation_dict[clean_word][0]
            syllable_count = len([phone for phone in pronunciation if phone[-1].isdigit()])
            return max(1, syllable_count)
        
        # Fallback: simple vowel counting method
        return self._count_syllables_fallback(clean_word)
    
    def _count_syllables_fallback(self, word: str) -> int:
        """
        Fallback syllable counting using vowel patterns.
        
        Args:
            word: Clean word to analyze
            
        Returns:
            Estimated syllable count (minimum 1)
        """
        # Remove common silent endings
        word = re.sub(r'[^aeiouy]*e$', '', word.lower())
        
        # Count vowel groups
        vowel_groups = re.findall(r'[aeiouy]+', word)
        syllable_count = len(vowel_groups)
        
        # Handle special cases
        if word.endswith('le') and len(word) > 2 and word[-3] not in 'aeiouy':
            syllable_count += 1
        
        return max(1, syllable_count)
    
    def analyze_text(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Analyze text and calculate Flesch-Kincaid grade level.
        
        Args:
            text: The text to analyze
            
        Returns:
            Dictionary with analysis results or None if analysis fails
        """
        if not text or not text.strip():
            logger.warning("Empty or whitespace-only text provided for analysis")
            return None
        
        try:
            # Clean and split into sentences
            sentences = self._extract_sentences(text)
            if not sentences:
                logger.warning("No sentences found in text")
                return None
            
            # Extract words
            words = self._extract_words(text)
            if not words:
                logger.warning("No words found in text")
                return None
            
            # Count syllables
            total_syllables = sum(self.count_syllables(word) for word in words)
            
            # Calculate metrics
            total_words = len(words)
            total_sentences = len(sentences)
            
            avg_sentence_length = total_words / total_sentences
            avg_syllables_per_word = total_syllables / total_words
            
            # Calculate Flesch-Kincaid Grade Level
            grade_level = (0.39 * avg_sentence_length) + (11.8 * avg_syllables_per_word) - 15.59
            
            # Ensure grade level is not negative
            grade_level = max(0, grade_level)
            
            result = {
                'grade_level': round(grade_level, 2),
                'total_words': total_words,
                'total_sentences': total_sentences,
                'total_syllables': total_syllables,
                'avg_sentence_length': round(avg_sentence_length, 2),
                'avg_syllables_per_word': round(avg_syllables_per_word, 2)
            }
            
            logger.debug(f"Text analysis complete: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing text: {e}")
            return None
    
    def _extract_sentences(self, text: str) -> List[str]:
        """Extract sentences from text."""
        # Split on sentence-ending punctuation
        sentences = re.split(r'[.!?]+', text)
        # Filter out empty sentences and strip whitespace
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences
    
    def _extract_words(self, text: str) -> List[str]:
        """Extract words from text."""
        # Find word boundaries, excluding punctuation
        words = re.findall(r'\b\w+\b', text.lower())
        return words
    
    def analyze_story_pages(self, pages: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        Analyze readability for a complete story across multiple pages.
        
        Args:
            pages: List of page dictionaries containing 'text' field
            
        Returns:
            Dictionary with analysis results or None if analysis fails
        """
        if not pages:
            logger.warning("No pages provided for story analysis")
            return None
        
        # Combine all page text
        page_texts = []
        for page in pages:
            if isinstance(page, dict) and 'text' in page and page['text']:
                page_texts.append(page['text'])
        
        if not page_texts:
            logger.warning("No text content found in story pages")
            return None
        
        full_text = " ".join(page_texts)
        
        logger.info(f"Analyzing story with {len(pages)} pages, {len(full_text)} characters")
        
        return self.analyze_text(full_text)
    
    def get_difficulty_level_info(self, difficulty_level: int) -> Dict[str, Any]:
        """
        Get information about a specific difficulty level.
        
        Args:
            difficulty_level: Integer difficulty level (0-10)
            
        Returns:
            Dictionary with level information
        """
        level_info = {
            0: {"name": "Level 0", "age_range": "3-4 years", "description": "Pre-reading"},
            1: {"name": "Level 1", "age_range": "5 years", "description": "Beginning sounds"},
            2: {"name": "Level 2", "age_range": "6 years", "description": "Early reading"},
            3: {"name": "Level 3", "age_range": "7 years", "description": "Developing reading"},
            4: {"name": "Level 4", "age_range": "8 years", "description": "Fluent reading"},
        }
        
        return level_info.get(difficulty_level, {
            "name": f"Level {difficulty_level}",
            "age_range": "Unknown",
            "description": "Future expansion"
        })


# Global instance for use across the application
readability_analyzer = ReadabilityAnalyzer()
