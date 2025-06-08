"""
Sample stories for each difficulty level to use in prompts.
These examples help the LLM understand the target complexity.
"""

SAMPLE_STORIES = {
    0: {
        "title": "The Cat",
        "text": "Sam has a cat. The cat is big. The cat is red. Sam likes his cat. The cat likes Sam.",
        "description": "Very short sentences (3-5 words), simple 1-syllable words, basic concepts"
    },
    1: {
        "title": "The Dog's Ball", 
        "text": "I have a dog. His name is Max. Max likes to play. He has a red ball. We play in the yard. Max runs fast. I throw the ball. Max brings it back.",
        "description": "Short sentences (4-8 words), mostly 1-2 syllable words, simple stories"
    },
    2: {
        "title": "The Little Bird",
        "text": "There was a little blue bird. She lived in a tall tree. Every morning she would sing. The other animals liked her songs. One day she lost her voice. A kind rabbit helped her find some honey. Soon she could sing again.",
        "description": "Medium sentences (6-12 words), mix of 1-3 syllable words, simple plot"
    },
    3: {
        "title": "The Magic Garden",
        "text": "Emma found a garden behind her house. The flowers were pretty colors. Red roses grew by yellow flowers. She saw a shiny leaf on a tree. It looked like it had stars on it. Emma liked to visit the garden after school.",
        "description": "Longer sentences (8-15 words), some complex words, more detailed descriptions"
    },
    4: {
        "title": "The Young Explorer",
        "text": "Alex wanted to be an explorer like his dad. He read books about far away places. In the summer he went camping with his family. There he learned to use a compass. He also learned about trees and animals. Each trip made him want to see more places.",
        "description": "Varied sentence length (10-20 words), broader vocabulary, complex concepts"
    },
    5: {
        "title": "The Secret Laboratory",
        "text": "Maya found something cool in her grandmother's basement. Behind old boxes, she saw a small lab with interesting science things. There were glass bottles with bright colored water. Her grandmother said she used to be a scientist who studied rocks. Maya thought science was just as fun as reading stories.",
        "description": "Longer sentences (12-25 words), advanced vocabulary, scientific concepts"
    },
    6: {
        "title": "The Time Machine Adventure", 
        "text": "Emma found a special machine hidden in her uncle's workshop. When she turned on the device, it sent her back to old England. She met brave knights who lived in large stone castles. The dragons in this world looked completely real and very scary. Emma knew she had to be careful not to change what happened in history. She needed to find a safe way to return to her own time.",
        "description": "Complex sentences (15-30 words), sophisticated vocabulary, abstract concepts"
    },
    7: {
        "title": "The River Mystery",
        "text": "Jordan had always cared deeply about protecting the natural environment in her community. She noticed that the river flowing near her neighborhood was becoming polluted with dangerous waste products. Jordan decided to carefully investigate what was causing this serious environmental problem. She asked her science teacher to help her learn the proper methods for testing contaminated water samples. Her detailed analysis revealed that a local manufacturing company was illegally releasing toxic chemicals into the river system. Jordan presented her scientific findings to the city government officials who agreed to take immediate action.",
        "description": "Advanced sentence structure (20-35 words), academic vocabulary, real-world issues"
    },
    8: {
        "title": "The Science Project",
        "text": "Marcus loved learning about how tiny pieces of matter work together. For the school science fair, he wanted to show how small particles can connect to each other. His physics teacher helped him build a special machine with lights and sensors. The project was hard work, but Marcus was able to show how particles stay connected even when far apart. The college teachers were amazed by his smart work and told him to keep studying science.",
        "description": "Complex academic language (25-40 words), technical vocabulary, advanced scientific concepts"
    }
}

def get_sample_story(level: int) -> dict:
    """Get sample story for a specific difficulty level."""
    return SAMPLE_STORIES.get(level, SAMPLE_STORIES[2])  # Default to level 2

def get_all_samples() -> dict:
    """Get all sample stories."""
    return SAMPLE_STORIES 