import numpy as np
from PIL import Image


class ZImageClient:
    def __init__(self, config: dict):
        self.dry_run = config.get("dry_run", False)
        self.model_id = config["model_id"]
        self.hf_token = config.get("hf_token")
        self.width = config.get("width", 1024)
        self.height = config.get("height", 1024)
        self._client = None

    @property
    def client(self):
        if self._client is None and not self.dry_run:
            from huggingface_hub import InferenceClient
            self._client = InferenceClient(token=self.hf_token)
        return self._client

    def generate(self, prompt: str) -> Image.Image:
        if self.dry_run:
            arr = np.random.randint(50, 200, (self.height, self.width, 3), dtype=np.uint8)
            return Image.fromarray(arr)
        return self.client.text_to_image(
            prompt,
            model=self.model_id,
            width=self.width,
            height=self.height,
        )
