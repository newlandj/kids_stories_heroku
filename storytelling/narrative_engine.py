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

# Local imports
from utils.content_safety import ContentScreener
from settings import AppConfig

# Configure logging
logger = logging.getLogger("kids-story-lambda")

import asyncio

class FableFactory:
    
    async def generate_story_package(self, prompt: str) -> dict:
        """
        Orchestrates the creation of all story components concurrently.
        Steps:
            1. Generate narrative text
            2. Identify key scenes
            3. Generate illustrations for each scene (async)
            4. Generate narration for each scene (async)
            5. Gather results and return a complete story package
        """
        # Step 1: Generate narrative text
        story_text = await self.weave_narrative(prompt)
        # Step 2: Identify key scenes
        scenes = self._identify_key_scenes(story_text)
        # Step 3: Generate illustrations for each scene (async)
        illustration_tasks = []
        for idx, scene in enumerate(scenes):
            art_direction = self._generate_art_direction(scene, idx) if hasattr(self, '_generate_art_direction') else ""
            illustration_tasks.append(self._generate_single_illustration(scene, idx, art_direction))
        visual_elements = await asyncio.gather(*illustration_tasks)

        # Step 4: Generate narration for each scene (async)
        audio_narration = await self.record_narration(scenes)

        # TODO: Step 5: Gather results and return a complete story package
        word_count = len(story_text.split())
        illustration_count = len(visual_elements)
        scene_count = len(scenes)
        return {
            "story_text": story_text,
            "scenes": scenes,
            "visual_elements": visual_elements,
            "audio_narration": audio_narration,
            "word_count": word_count,
            "illustration_count": illustration_count,
            "scene_count": scene_count
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
        skip_db = os.environ.get("SKIP_DB", "false").lower() in ("true","1","yes")
        self._is_aws_env = False if skip_db else bool(os.environ.get("AWS_LAMBDA_FUNCTION_NAME"))
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
        # Fetch API key: SSM in AWS or env var locally
        if os.environ.get("AWS_LAMBDA_FUNCTION_NAME"):
            param = AppConfig.get_value("openai_ssm_param_name")
            if not param:
                logger.error("SSM parameter name not configured.")
                return None
            ssm = boto3.client('ssm')
            try:
                resp = ssm.get_parameter(Name=param, WithDecryption=True)
                return resp['Parameter']['Value']
            except Exception as e:
                logger.error(f"Failed to fetch OpenAI key: {e}")
                return None
        return os.environ.get("OPENAI_API_KEY")

    async def weave_narrative(self, prompt: str) -> str:
        if os.environ.get("SKIP_DB", "false").lower() in ("true", "1", "yes"):
            return "Once upon a time, there was a brave child who went on an adventure."
        self.content_screener.validate_prompt(prompt)
        style = self._analyze_theme_for_style(prompt)
        directive = self._create_storyteller_directive(style)
        attempts, backoff = 3, 1
        for i in range(attempts):
            try:
                resp = await asyncio.to_thread(
                    self.client.chat.completions.create,
                    model="gpt-4o",
                    messages=[{"role":"system","content":directive},
                              {"role":"user","content":f"Create a children's story about: {prompt}"}],
                    temperature=0.7 + i*0.1,
                    max_tokens=1500,
                    presence_penalty=0.6,
                    frequency_penalty=0.5
                )
                text = resp.choices[0].message.content
                return self._format_story(text) if self._validate_story_quality(text) else text
            except Exception as e:
                logger.error(f"weave_narrative error {i}: {e}")
                if i < attempts-1:
                    await asyncio.sleep(backoff); backoff *= 2
        # Fallback
        return self._generate_fallback_story(prompt)

    def _identify_key_scenes(self, story_text: str) -> List[str]:
        if os.environ.get("SKIP_DB", "false").lower() in ("true", "1", "yes"):
            return ["Scene 1: The adventure begins.", "Scene 2: The challenge.", "Scene 3: The resolution."]
        paras = [p.strip() for p in story_text.replace("\r","\n").split("\n\n") if p.strip()]
        if len(paras) <=4: return paras
        markers=[]
        for idx,p in enumerate(paras):
            if p.startswith('"') or any(w in p.lower() for w in ["suddenly","later","meanwhile"]): markers.append(idx)
        if 2<=len(markers)<=5:
            scenes=[]; prev=0
            for m in markers:
                if m>prev: scenes.append(" ".join(paras[prev:m])); prev=m
            if prev<len(paras): scenes.append(" ".join(paras[prev:]))
            if 2<=len(scenes)<=5: return scenes
        # fallback equal segments
        n=4; size=max(1,len(paras)//n); scenes=[" ".join(paras[i:i+size]) for i in range(0,len(paras),size)][:n]
        return scenes

    async def _generate_single_illustration(self, scene, idx, art_direction):
        if os.environ.get("SKIP_DB", "false").lower() in ("true", "1", "yes"):
            return f"dummy-illustration-{idx}.png"
        prompt = f"Colorful children's book illustration, {art_direction}: {scene}" 
        resp = await asyncio.to_thread(self.client.images.generate, model="dall-e-3", prompt=prompt, size="1024x1024", n=1)
        url=resp.data[0].url
        return await asyncio.to_thread(self._compress_and_store_image, url, idx)

    async def record_narration(self, scenes):
        if os.environ.get("SKIP_DB", "false").lower() in ("true", "1", "yes"):
            return [ {"audio_url":"dummy-narration.mp3","scene_index":i} for i,_ in enumerate(scenes) ]
        voice = self._select_voice_for_story(scenes[0] if scenes else "")
        tasks=[asyncio.create_task(self._generate_single_narration(scene,i,voice)) for i,scene in enumerate(scenes) if scene]
        return await asyncio.gather(*tasks)

def generate_story_package(prompt: str) -> dict:
    """
    Synchronous wrapper for Lambda compatibility.
    Orchestrates story generation using FableFactory.
    """
    factory = FableFactory()
    return asyncio.run(factory.generate_story_package(prompt))

# All logic now lives in FableFactory methods above.
