from pathlib import Path
from typing import TypedDict, Annotated, Sequence
from PIL import Image
from langgraph.graph import StateGraph, END


class PipelineState(TypedDict):
    raw_description: str
    image_prompt: str
    video_prompt: str
    generated_image: Image.Image
    video_frames: list[Image.Image]
    ply_path: Path
    output_mesh_path: Path
    error: str


def _inject(model_key: str, config: dict) -> dict:
    cfg = dict(config["models"][model_key])
    if config.get("dry_run", False):
        cfg["dry_run"] = True
    if hf_token := config.get("hf_token"):
        cfg["hf_token"] = hf_token
    return cfg


def create_pipeline(config: dict) -> StateGraph:
    from pipeline.post_processor import PostProcessor
    from models.glm_client import GLMClient
    from models.zimage_client import ZImageClient
    from models.ltx_client import LTXClient
    from models.hunyuan_client import HunyuanWorldMirrorClient

    glm = GLMClient(_inject("glm", config))
    zimage = ZImageClient(_inject("zimage", config))
    ltx = LTXClient(_inject("ltx", config))
    hunyuan = HunyuanWorldMirrorClient(_inject("hunyuan", config))
    post = PostProcessor(config)

    output_dir = Path(config["output"]["directory"])

    def step_prompt_optimization(state: PipelineState) -> dict:
        print("[1/5] Generating prompts...")
        prompts = glm.optimize_prompts(state["raw_description"])
        return {
            "image_prompt": prompts.image_prompt,
            "video_prompt": prompts.video_prompt,
        }

    def step_image_generation(state: PipelineState) -> dict:
        print("[2/5] Generating image with Z-Image...")
        image = zimage.generate(state["image_prompt"])
        return {"generated_image": image}

    def step_video_generation(state: PipelineState) -> dict:
        print("[3/5] Generating video with LTX-2...")
        frames = ltx.generate(state["generated_image"], state["video_prompt"])
        return {"video_frames": frames}

    def step_3d_generation(state: PipelineState) -> dict:
        print("[4/5] Generating 3D scene with HunyuanWorld-Mirror...")
        scene_dir = output_dir / "hunyuan_output"
        ply = hunyuan.generate(state["video_frames"], scene_dir)
        return {"ply_path": ply}

    def step_post_processing(state: PipelineState) -> dict:
        print("[5/5] Post-processing with Open3D...")
        mesh_output = output_dir / "final_mesh"
        fbx = post.process(state["ply_path"], mesh_output)
        return {"output_mesh_path": fbx}

    builder = StateGraph(PipelineState)

    builder.add_node("prompt_optimization", step_prompt_optimization)
    builder.add_node("image_generation", step_image_generation)
    builder.add_node("video_generation", step_video_generation)
    builder.add_node("3d_generation", step_3d_generation)
    builder.add_node("post_processing", step_post_processing)

    builder.set_entry_point("prompt_optimization")
    builder.add_edge("prompt_optimization", "image_generation")
    builder.add_edge("image_generation", "video_generation")
    builder.add_edge("video_generation", "3d_generation")
    builder.add_edge("3d_generation", "post_processing")
    builder.add_edge("post_processing", END)

    return builder.compile()
