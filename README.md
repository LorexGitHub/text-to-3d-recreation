# Text-to-3D Pipeline

LangGraph pipeline that generates explorable 3D worlds from text descriptions.

```
Raw Text → [GLM-4.7 / Templates] → Prompts → [Z-Image / FLUX via HF API] → Image → [LTX-2 via HF API] → Video → [HunyuanWorld-Mirror] → PLY → [Open3D] → Textured Mesh (OBJ)
```

## Quick Start

```bash
# Dry run (configured in config.yaml, true gives no API calls, generates dummy output)
python main.py "a cozy living room with a fireplace"
```

## Configuration

Edit `config.yaml`. Set `dry_run: false` and add your HF token for real model calls.

| Step | Provider | Requires |
|------|----------|----------|
| Prompts | Local templates (or LM Studio) | Nothing |
| Z-Image | HuggingFace Inference API | HF token, credits |
| LTX-2 | HuggingFace Inference API | HF token, credits |
| 3D | HuggingFace Gradio Space (free) | HF token |
| Mesh | Open3D (local CPU) | Nothing |

## Output

`output/final_mesh/scene.obj` — textured mesh + `texture.png`
