import logging
import random
import re
from typing import Any, Dict

logger = logging.getLogger("kids-story-app")


class ContentScreener:
    """Screens content to ensure it's appropriate for children"""

    def __init__(self):
        # Keywords that might indicate inappropriate content
        self._inappropriate_terms = [
            "violent",
            "kill",
            "murder",
            "blood",
            "death",
            "weapon",
            "gun",
            "knife",
            "suicide",
            "drug",
            "alcohol",
            "naked",
            "nude",
            "sex",
            "explicit",
            "terror",
            "nightmare",
            "horror",
            "gruesome",
            # Additional terms would be included in a real implementation
        ]

        # Compile regex patterns for efficiency
        self._inappropriate_pattern = re.compile(
            r"\b("
            + "|".join(re.escape(term) for term in self._inappropriate_terms)
            + r")\b",
            re.IGNORECASE,
        )

        # Topics appropriate for children's stories
        self._appropriate_topics = [
            "friendship",
            "adventure",
            "animals",
            "family",
            "learning",
            "imagination",
            "nature",
            "kindness",
            "helping",
            "sharing",
            "courage",
            "creativity",
            "discovery",
            "growing up",
            "teamwork",
        ]

    def validate_prompt(self, prompt: str) -> None:
        """
        Validates that a prompt is appropriate for children's content.
        Raises an exception if inappropriate content is detected.
        """
        # Check for inappropriate terms
        matches = self._inappropriate_pattern.findall(prompt.lower())

        if matches:
            # Log the issue but don't expose specific terms in the error
            logger.warning(f"Inappropriate content detected in prompt: {matches}")
            raise ValueError(
                "Your prompt contains themes that aren't appropriate for children's stories. "
                "Please provide a prompt with child-friendly themes like friendship, adventure, "
                "or animals."
            )

    def suggest_alternative(self, prompt: str) -> str:
        """
        Suggests an alternative child-friendly topic if the original seems inappropriate.
        Used as a fallback when content moderation is triggered.
        """
        # Extract any positive themes from the prompt
        positive_themes = [
            word for word in prompt.lower().split() if word in self._appropriate_topics
        ]

        if positive_themes:
            # Use positive themes from the original prompt
            theme = random.choice(positive_themes)
            return f"a story about {theme}"
        else:
            # Suggest a completely new theme
            theme = random.choice(self._appropriate_topics)
            return f"a story about {theme}"

    def screen_generated_content(self, content: str) -> Dict[str, Any]:
        """
        Screens generated content to ensure it's appropriate.
        Returns a dict with 'is_appropriate' flag and optional 'reason'.
        """
        # Check for inappropriate terms
        matches = self._inappropriate_pattern.findall(content.lower())

        if matches:
            logger.warning("Inappropriate content detected in generated story")
            return {
                "is_appropriate": False,
                "reason": "Generated content contains inappropriate themes for children",
            }

        # Additional content checks could be implemented here

        return {"is_appropriate": True}
