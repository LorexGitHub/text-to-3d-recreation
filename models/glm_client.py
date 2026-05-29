from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from dataclasses import dataclass


@dataclass
class OptimizedPrompts:
    image_prompt: str
    video_prompt: str


IMAGE_SYSTEM_PROMPT = """You are a prompt engineering assistant for a 3D world generation pipeline.
Given a room description, produce a single concise prompt for Z-Image (text-to-image model).

Rules:
- Focus on visual details: lighting, colors, materials, architecture, layout
- Specify style (e.g., photorealistic, wide-angle)
- Keep it under 100 words
- Do NOT mention cameras, rotations, or movement
- Output ONLY the prompt text, no labels"""

VIDEO_SYSTEM_PROMPT = """You are a prompt engineering assistant for a 3D world generation pipeline.
Given a room description, produce a single concise prompt for LTX-2 (image-to-video model).

Rules:
- Describe a smooth 360-degree camera rotation around the vertical axis
- The camera starts at the exact geometric center of the room at eye height
- Emphasize constant speed, no acceleration, no wobble, no lateral movement
- The rotation completes one full uninterrupted circle
- No motion blur distortion, no environmental deformation
- Use cinematic language
- Output ONLY the prompt text, no labels"""


def make_template_prompts(raw_description: str) -> OptimizedPrompts:
    image_prompt = (
        f"A photorealistic, wide-angle architectural shot of {raw_description}. "
        f"High definition, 8k resolution, cinematic warm lighting. "
        f"Sharp geometric lines, clean minimalist atmosphere."
    )
    video_prompt = (
        f"The scene begins with the camera fixed at the exact geometric center "
        f"of the room described as: {raw_description}. "
        f"The camera immediately initiates a perfectly smooth, constant-speed "
        f"360-degree rotation around its vertical axis with no acceleration, "
        f"no deceleration, and no lateral movement. The camera remains fully "
        f"stabilized, level, and locked at eye height, rotating like a "
        f"precision-controlled gimbal to reveal each wall sequentially in "
        f"equal intervals. The rotation completes one uninterrupted full circle "
        f"with no camera shake, no motion blur distortion, and no environmental deformation."
    )
    return OptimizedPrompts(image_prompt=image_prompt, video_prompt=video_prompt)


class GLMClient:
    def __init__(self, config: dict):
        self.use_templates = config.get("use_templates", True)
        if not self.use_templates:
            api_base = config["api_base"]
            api_key = config.get("api_key", "not-needed")
            model_name = config["model_name"]
            temperature = config.get("temperature", 0.3)
            max_tokens = config.get("max_tokens", 2048)

            self.llm = ChatOpenAI(
                base_url=api_base,
                api_key=api_key,
                model=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
            )

    def optimize_prompts(self, raw_description: str) -> OptimizedPrompts:
        if self.use_templates:
            return make_template_prompts(raw_description)
        image_prompt = self._generate(self._build_messages(IMAGE_SYSTEM_PROMPT, raw_description))
        video_prompt = self._generate(self._build_messages(VIDEO_SYSTEM_PROMPT, raw_description))
        return OptimizedPrompts(image_prompt=image_prompt.strip(), video_prompt=video_prompt.strip())

    def _build_messages(self, system: str, description: str) -> list:
        return [SystemMessage(content=system), HumanMessage(content=description)]

    def _generate(self, messages: list) -> str:
        response = self.llm.invoke(messages)
        return response.content
