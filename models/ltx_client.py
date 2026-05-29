import os
import tempfile
from PIL import Image
import numpy as np


def _frames_from_mp4(video_bytes: bytes, max_frames: int | None = None) -> list[Image.Image]:
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    try:
        tmp.write(video_bytes)
        tmp.flush()
        tmp.close()
        try:
            import imageio.v3 as iio
            arr = iio.imread(tmp.name)
            if arr.ndim == 4:
                frames = [Image.fromarray(f) for f in arr]
            else:
                frames = [Image.fromarray(arr)]
        except ImportError:
            try:
                import cv2
                cap = cv2.VideoCapture(tmp.name)
                frames = []
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
                cap.release()
            except ImportError:
                raise ImportError(
                    "Cannot extract video frames. Install: pip install imageio[ffmpeg]"
                )
    finally:
        os.unlink(tmp.name)
    if max_frames:
        frames = frames[:max_frames]
    return frames


class LTXClient:
    def __init__(self, config: dict):
        self.dry_run = config.get("dry_run", False)
        self.model_id = config["model_id"]
        self.hf_token = config.get("hf_token")
        self.width = config.get("width", 512)
        self.height = config.get("height", 512)
        self.num_frames = config.get("num_frames", 16)
        self._client = None

    @property
    def client(self):
        if self._client is None and not self.dry_run:
            from huggingface_hub import InferenceClient
            self._client = InferenceClient(token=self.hf_token)
        return self._client

    def generate(self, image: Image.Image, prompt: str) -> list[Image.Image]:
        if self.dry_run:
            n = min(self.num_frames, 16)
            arr = np.array(image.resize((self.width, self.height)))
            return [Image.fromarray(np.roll(arr, shift=i * 10, axis=1)) for i in range(n)]
        video_bytes = self.client.image_to_video(
            image=image,
            model=self.model_id,
            prompt=prompt,
            num_frames=self.num_frames,
        )
        frames = _frames_from_mp4(video_bytes, max_frames=self.num_frames)
        return frames
