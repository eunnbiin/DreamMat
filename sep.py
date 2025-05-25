import trimesh
import numpy as np

TARGET_IDX = [2, 5, 8, 13]
ORIGINAL_GLB_PATH = "/threestudio_dreammat/load/shapes/objs/knight.glb"
OUTPUT_DIR = "/threestudio_dreammat/load/shapes/objs/"

mesh = trimesh.load(ORIGINAL_GLB_PATH)
if isinstance(mesh, trimesh.Scene):
    mesh = trimesh.util.concatenate(mesh.dump())
elif not isinstance(mesh, trimesh.Trimesh):
    try:
        mesh = trimesh.util.concatenate(mesh)
    except Exception as e:
        print(f"Warning: Could not convert loaded object to single Trimesh: {e}")

mat = np.load("/threestudio_dreammat/load/shapes/sep/mesh_0.0.npy")

all_target_indices = []
for target_idx in TARGET_IDX:
    target_indices = np.where(mat == target_idx)[0]
    all_target_indices.extend(target_indices)

all_target_indices = np.unique(all_target_indices)
max_face_idx = len(mesh.faces) - 1
valid_target_indices = all_target_indices[all_target_indices <= max_face_idx]
ㄷ
all_face_indices = np.arange(len(mesh.faces))
remaining_indices = np.setdiff1d(all_face_indices, valid_target_indices)

idx_str = "_".join(map(str, TARGET_IDX))

if len(valid_target_indices) > 0:
    target_submesh = mesh.submesh([valid_target_indices], append=True)
    target_output_path = f"{OUTPUT_DIR}knight_target_{idx_str}.glb"
    target_submesh.export(target_output_path)
    print(f"타겟 메쉬 저장: {target_output_path}")

if len(remaining_indices) > 0:
    remaining_submesh = mesh.submesh([remaining_indices], append=True)
    remaining_output_path = f"{OUTPUT_DIR}knight_remaining_{idx_str}.glb"
    remaining_submesh.export(remaining_output_path)
    print(f"나머지 메쉬 저장: {remaining_output_path}")

print(f"타겟 face 개수: {len(valid_target_indices)}")
print(f"나머지 face 개수: {len(remaining_indices)}")
print(f"전체 face 개수: {len(mesh.faces)}")