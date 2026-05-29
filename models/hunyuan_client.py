import os
from pathlib import Path
from PIL import Image


def _file_data(path: str) -> dict:
    return {
        "path": os.path.abspath(path),
        "meta": {"_type": "gradio.FileData"},
        "orig_name": os.path.basename(path),
    }


class HunyuanWorldMirrorClient:
    def __init__(self, config: dict):
        self.dry_run = config.get("dry_run", False)
        self.space_id = config.get("space_id", "tencent/HunyuanWorld-Mirror")
        self.hf_token = config.get("hf_token")
        self.frame_stride = config.get("frame_stride", 2)
        self.max_frames = config.get("max_frames", 30)
        self._client = None

    @property
    def client(self):
        if self._client is None and not self.dry_run:
            from gradio_client import Client
            self._client = Client(self.space_id, token=self.hf_token, verbose=False)
        return self._client

    def generate(self, video_frames: list[Image.Image], output_dir: str | Path) -> Path:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        ply_path = output_dir / "scene.ply"

        if self.dry_run:
            self._make_dummy_ply(ply_path)
            return ply_path

        try:
            frames_dir = output_dir / "space_input_frames"
            frames_dir.mkdir(exist_ok=True)

            sampled = video_frames[::self.frame_stride][:self.max_frames]
            img_paths = []
            for i, frame in enumerate(sampled):
                p = frames_dir / f"frame_{i:04d}.png"
                frame.save(p)
                img_paths.append(str(p))

            result = self.client.predict(
                [_file_data(p) for p in img_paths],
                1.0,
                api_name="/update_gallery_on_file_upload",
            )

            workspace_path = ""
            if isinstance(result, (list, tuple)):
                for item in result:
                    if isinstance(item, str) and len(item) > 5:
                        workspace_path = item
                        break
            elif isinstance(result, str):
                workspace_path = result

            if not workspace_path:
                raise RuntimeError("no workspace path returned from file upload")

            result = self.client.predict(
                workspace_path,
                "All",
                True,
                False,
                True,
                False,
                api_name="/gradio_demo",
            )

            model3d_path = None
            flat = result if isinstance(result, (list, tuple)) else [result]
            for item in flat:
                if isinstance(item, str) and item.lower().endswith((".ply", ".obj", ".glb")):
                    model3d_path = item
                    break
                if isinstance(item, dict):
                    for key in ("path", "value", "file", "name"):
                        val = item.get(key)
                        if isinstance(val, str) and val.lower().endswith((".ply", ".obj", ".glb")):
                            model3d_path = val
                            break

            if model3d_path and os.path.exists(model3d_path):
                import shutil
                shutil.copy(model3d_path, ply_path)
                print(f"Downloaded 3D scene from Space to {ply_path}")
            else:
                print(f"No model file found in Space response, using dummy PLY")
                self._make_dummy_ply(ply_path)

        except Exception as e:
            print(f"Space API call failed ({e}), falling back to dummy PLY")
            self._make_dummy_ply(ply_path)

        return ply_path

    def _make_dummy_ply(self, path: Path) -> None:
        import open3d as o3d
        import numpy as np
        mesh = o3d.geometry.TriangleMesh.create_box(width=2.0, height=2.0, depth=2.0)
        mesh.translate(-mesh.get_center())
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.asarray(mesh.vertices))
        dense = pcd.voxel_down_sample(0.05)
        dense_pts = np.asarray(dense.points) + np.random.uniform(-0.02, 0.02, np.asarray(dense.points).shape)
        dense.points = o3d.utility.Vector3dVector(dense_pts)
        colors = np.tile([0.6, 0.6, 0.8], (len(dense_pts), 1))
        dense.colors = o3d.utility.Vector3dVector(colors)
        o3d.io.write_point_cloud(str(path), dense)
