"""
AI pipeline logic:
- USE_DUMMY_AI=true: Use dummy images/audio (for fast local/dev testing, no API calls)
"""
import os
import logging
import random
from typing import List, Optional
import time
import asyncio
import json

# For API operations
import openai
from openai import OpenAI
import httpx

# For file operations
import io
import tempfile
import boto3
from urllib.parse import urlparse
from PIL import Image
import requests
from app.storage import upload_file_to_s3

# Local imports
from app.content_safety import ContentScreener
from app.settings import AppConfig

# Configure logging
logger = logging.getLogger("kids-story-lambda")

import asyncio

class FableFactory:
    
    def __init__(self):
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

    async def generate_story_package(self, prompt: str) -> dict:
        """
        Orchestrates the generation of children's story elements using AI (structured JSON pipeline)
        """
        start_time = time.monotonic()
        story = await self.weave_narrative(prompt)  # story is now a dict with title, characters, pages
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

        # Wire imageUrl and audioUrl into each page
        for idx, page in enumerate(pages):
            if idx < len(visual_elements):
                page["imageUrl"] = visual_elements[idx]
            # Find matching audio narration by index
            audio = next((a for a in audio_narration if a.get("page_index") == idx), None)
            if audio:
                page["audioUrl"] = audio["audio_url"]

        word_count = sum(len(page["text"].split()) for page in pages)
        illustration_count = len(visual_elements)
        page_count = len(pages)
        elapsed = time.monotonic() - start_time
        logger.info(f"Story generation complete in {elapsed:.2f} seconds. Title: {story.get('title')}, {page_count} pages, {illustration_count} images, {word_count} words.")
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

    async def weave_narrative(self, prompt: str) -> dict:
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

        system_prompt = (
            "You are a children's story writer creating engaging, age-appropriate stories for children in grades 1-3 (ages 6-9). "
            "Create stories with 3-4 pages, each with a vivid scene that can be illustrated. For each story, create detailed character descriptions that should remain consistent throughout the story. For each page, provide a detailed image description that maintains character consistency. "
            "IMPORTANT: Each story MUST include a creative, appropriate book title, and the title MUST be returned as the 'title' field in the structured JSON output."
        )
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

    async def _generate_single_illustration(self, page_prompt, idx, art_direction):
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
        Selects a TTS voice based on story keywords.
        - Adventure/brave/exciting: 'alloy'
        - Sweet/gentle/kind: 'nova'
        - Mystery/dark: 'echo'
        - Default: 'onyx'
        """
        t = text.lower() if text else ""
        if any(word in t for word in ["adventure", "brave", "exciting", "explore", "journey"]):
            return "alloy"
        if any(word in t for word in ["sweet", "gentle", "kind", "love", "friend"]):
            return "nova"
        if any(word in t for word in ["mystery", "dark", "secret", "shadow", "spooky"]):
            return "echo"
        return "onyx"

    async def _generate_single_narration(self, page_text, idx, voice):
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
