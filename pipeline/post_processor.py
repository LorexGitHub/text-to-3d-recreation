from pathlib import Path
import numpy as np
import open3d as o3d


class PostProcessor:
    def __init__(self, config: dict):
        pp = config["post_processing"]
        self.nb_neighbors = pp["outlier_removal"]["nb_neighbors"]
        self.std_ratio = pp["outlier_removal"]["std_ratio"]
        self.poisson_depth = pp["poisson"]["depth"]
        self.simplify_ratio = pp["simplify"]["target_percentage"]
        self.texture_res = pp["texture"]["resolution"]

    def process(self, ply_path: Path, output_dir: Path) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        pcd = o3d.io.read_point_cloud(str(ply_path))

        pcd = self._remove_outliers(pcd)
        pcd = self._downsample(pcd)
        mesh = self._poisson_reconstruction(pcd)
        mesh = self._simplify_mesh(mesh)
        texture_path = self._bake_texture(mesh, pcd, output_dir)
        fbx_path = self._export_fbx(mesh, texture_path, output_dir)
        return fbx_path

    def _remove_outliers(self, pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
        cl, _ = pcd.remove_statistical_outlier(
            nb_neighbors=self.nb_neighbors, std_ratio=self.std_ratio
        )
        return cl

    def _downsample(self, pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
        return pcd.voxel_down_sample(voxel_size=0.02)

    def _poisson_reconstruction(self, pcd: o3d.geometry.PointCloud) -> o3d.geometry.TriangleMesh:
        if not pcd.has_normals():
            pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
            )
            pcd.orient_normals_consistent_tangent_plane(100)

        mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=self.poisson_depth
        )
        bbox = pcd.get_axis_aligned_bounding_box()
        mesh = mesh.crop(bbox)
        return mesh

    def _simplify_mesh(self, mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:
        target = int(len(mesh.triangles) * self.simplify_ratio)
        mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=target)
        mesh = mesh.filter_smooth_simple(number_of_iterations=3)
        mesh.compute_vertex_normals()
        return mesh

    def _bake_texture(
        self, mesh: o3d.geometry.TriangleMesh, pcd: o3d.geometry.PointCloud, output_dir: Path
    ) -> Path:
        import xatlas
        import cv2
        from PIL import Image

        vertices = np.asarray(mesh.vertices)
        triangles = np.asarray(mesh.triangles)

        vmapping, indices, uvs = xatlas.parametrize(vertices, triangles)
        uv_map = (uvs * (self.texture_res - 1)).astype(np.uint32)
        texture = np.zeros((self.texture_res, self.texture_res, 3), dtype=np.uint8)
        counts = np.zeros((self.texture_res, self.texture_res), dtype=np.uint32)

        pcd_tree = o3d.geometry.KDTreeFlann(pcd)
        pcd_colors = np.asarray(pcd.colors) * 255

        for tri_idx in range(len(indices)):
            for i in range(3):
                atlas_vert = indices[tri_idx, i]
                orig_vert = vmapping[atlas_vert]
                u, v = uv_map[atlas_vert]
                if u < self.texture_res and v < self.texture_res:
                    pt = mesh.vertices[orig_vert]
                    _, nn_idx, _ = pcd_tree.search_knn_vector_3d(pt, 1)
                    color = pcd_colors[nn_idx[0]]
                    texture[v, u] = np.clip(color, 0, 255).astype(np.uint8)
                    counts[v, u] += 1

        mask = counts > 0
        for c in range(3):
            channel = texture[:, :, c].astype(np.float32)
            channel[mask] = (channel[mask] / counts[mask]).astype(np.uint8)
            texture[:, :, c] = np.clip(channel, 0, 255).astype(np.uint8)

        texture = self._inpaint_texture(texture, mask)

        tex_path = output_dir / "texture.png"
        Image.fromarray(texture).save(str(tex_path))
        return tex_path

    def _inpaint_texture(
        self, texture: np.ndarray, mask: np.ndarray
    ) -> np.ndarray:
        import cv2
        inpaint_mask = (1 - mask.astype(np.uint8)) * 255
        for _ in range(5):
            texture = cv2.inpaint(texture, inpaint_mask, 3, cv2.INPAINT_TELEA)
            mask = cv2.dilate(mask.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1) > 0
            inpaint_mask = (1 - mask.astype(np.uint8)) * 255
            if inpaint_mask.sum() == 0:
                break
        return texture

    def _export_fbx(
        self, mesh: o3d.geometry.TriangleMesh, texture_path: Path, output_dir: Path
    ) -> Path:
        obj_path = output_dir / "scene.obj"
        mtl_path = output_dir / "scene.mtl"

        mtl = f"""newmtl textureMat
map_Kd {texture_path.name}
illum 1
Kd 1.0 1.0 1.0
Ka 0.0 0.0 0.0
Ks 0.0 0.0 0.0
"""
        mtl_path.write_text(mtl)

        o3d.io.write_triangle_mesh(str(obj_path), mesh)
        obj_text = obj_path.read_text()
        obj_path.write_text(f"mtllib scene.mtl\n{obj_text}\nusemtl textureMat\n")
        fbx_path = output_dir / "scene.fbx"

        try:
            import bpy
            bpy.ops.wm.read_factory_settings(use_empty=True)
            bpy.ops.wm.obj_import(filepath=str(obj_path))
            bpy.ops.export_scene.fbx(filepath=str(fbx_path))
        except ImportError:
            pass

        return fbx_path if fbx_path.exists() else obj_path
