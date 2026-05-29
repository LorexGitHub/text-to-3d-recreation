# Text-to-3D Pipeline

LangGraph pipeline that generates explorable 3D worlds from text descriptions.

```
Raw Text → [GLM-4.7 / Templates] → Prompts → [Z-Image / FLUX via HF API] → Image → [LTX-2 via HF API] → Video → [HunyuanWorld-Mirror] → PLY → [Open3D] → Textured Mesh (OBJ)
```
## Showcase
Original Image:
<img width="1024" height="1024" alt="japanese_style_room" src="https://github.com/user-attachments/assets/bc7987bd-6ae4-4ecc-8485-bee9f4723f4d" />

Explorable 3D World/Object:
<img width="930" height="711" alt="image" src="https://github.com/user-attachments/assets/9d4e2b12-768f-4553-8f04-1540b7c2b438" />



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
