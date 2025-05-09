"""
AI pipeline logic:
- USE_DUMMY_AI=true: Use dummy images/audio (for fast local/dev testing, no API calls)
"""
import os
import logging
import random
from typing import List, Optional
import asyncio

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
    
    async def generate_story_package(self, prompt: str) -> dict:
        """
        Orchestrates the generation of children's story elements using AI (structured JSON pipeline)
        """
        import time
        start_time = time.monotonic()
        story = await self.weave_narrative(prompt)  # story is now a dict with title, characters, pages
        pages = story.get("pages", [])
        visual_elements = []
        audio_narration = []

        # Parallelize illustration and narration generation for all pages
        image_tasks = [
            self._generate_single_illustration(page["imagePrompt"], idx, art_direction=None)
            for idx, page in enumerate(pages)
        ]
        audio_tasks = [
            self._generate_single_narration(page["text"], idx, self._select_voice_for_story(page["text"]))
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
        # Removed story_structure_template loading to avoid missing method errors
        
    def _get_openai_api_key(self) -> Optional[str]:
        return os.environ.get("OPENAI_API_KEY")

    async def weave_narrative(self, prompt: str) -> dict:
        import json
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
        system_prompt = (
            "You are a children's story writer creating engaging, age-appropriate stories for children in grades 1-3 (ages 6-9). "
            "Create stories with 3-4 pages, each with a vivid scene that can be illustrated.\n\n"
            "For each story:\n"
            "1. Create detailed character descriptions that should remain consistent throughout the story\n"
            "2. For each page, provide a detailed image description that maintains character consistency\n"
            "3. Format the response as JSON with:\n"
            "   - title: string\n"
            "   - characters: array of {name: string, description: string}\n"
            "   - pages: array of {text: string, imagePrompt: string}\n"
            "Make character descriptions specific (e.g., 'a 7-year-old girl with curly red hair and green eyes, wearing a yellow polka dot dress')\n"
            "\nIMPORTANT: Respond ONLY with a valid JSON object. Do not include any commentary, explanation, or code fences."
        )
        user_message = f"Create a children's story about: {prompt}"
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
                    frequency_penalty=0.5
                )
                text = resp.choices[0].message.content
                # Try to extract JSON from the response
                try:
                    story_json = json.loads(text)
                except Exception:
                    # If not pure JSON, try to extract between code fences
                    import re
                    match = re.search(r"```json\\s*(.*?)```", text, re.DOTALL)
                    if match:
                        story_json = json.loads(match.group(1))
                    else:
                        # Fallback: try to parse as best as possible
                        story_json = json.loads(text[text.find('{'):text.rfind('}')+1])
                return story_json
            except Exception as e:
                logger.error(f"weave_narrative error {i}: {e}")
                if i < attempts-1:
                    await asyncio.sleep(backoff); backoff *= 2
        # Fallback
        return self._generate_fallback_story(prompt)

    def _identify_key_pages(self, story_text: str) -> List[str]:
        # Only use dummy page titles if USE_DUMMY_AI=true
        if os.environ.get("USE_DUMMY_AI", "false").lower() in ("true", "1", "yes"):
            return ["Page 1: The adventure begins.", "Page 2: The challenge.", "Page 3: The resolution."]
        paras = [p.strip() for p in story_text.replace("\r","\n").split("\n\n") if p.strip()]
        if len(paras) <=4: return paras
        markers=[]
        for idx,p in enumerate(paras):
            if p.startswith('"') or any(w in p.lower() for w in ["suddenly","later","meanwhile"]): markers.append(idx)
        if 2<=len(markers)<=5:
            pages=[]; prev=0
            for m in markers:
                if m>prev: pages.append(" ".join(paras[prev:m])); prev=m
            if prev<len(paras): pages.append(" ".join(paras[prev:]))
            if 2<=len(pages)<=5: return pages
        # fallback equal segments
        n=4; size=max(1,len(paras)//n); pages=[" ".join(paras[i:i+size]) for i in range(0,len(paras),size)][:n]
        return pages

    async def _generate_single_illustration(self, page_prompt, idx, art_direction):
        if os.environ.get("USE_DUMMY_AI", "false").lower() in ("true", "1", "yes"):
            return f"https://dummy-illustration-{idx}.png"
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
        import openai
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

    async def record_narration(self, pages):
        if os.environ.get("USE_DUMMY_AI", "false").lower() in ("true", "1", "yes"):
            return [ {"audio_url":"dummy-narration.mp3","page_index":i} for i,_ in enumerate(pages) ]
        voice = self._select_voice_for_story(pages[0] if pages else "")
        tasks=[asyncio.create_task(self._generate_single_narration(page_text,i,voice)) for i,page_text in enumerate(pages) if page_text]
        narrations = await asyncio.gather(*tasks)
        # narrations now only contain serializable fields (audio_url, page_index)
        return narrations

def generate_story_package(prompt: str) -> dict:
    """
    Synchronous wrapper for Lambda compatibility.
    Orchestrates story generation using FableFactory.
    """
    factory = FableFactory()
    return asyncio.run(factory.generate_story_package(prompt))

# All logic now lives in FableFactory methods above.
