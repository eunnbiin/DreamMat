# eunbin/project/github/DreamMat/threestudio_dreammat/threestudio/utils/mesh_utils.py

import trimesh
import numpy as np
import os
from typing import List, Tuple, Optional, Union, Any

def split_mesh_by_segmentation(
    original_mesh_path: str,
    segmentation_npy_path: str,
    target_segment_ids: List[int],
    output_dir: str,
    base_output_name: str = "submesh"
) -> Tuple[Optional[str], Optional[str]]:

    try:
        if not os.path.exists(original_mesh_path):
            print(f"Error: Original mesh file not found: {original_mesh_path}")
            return None, None

        # Load the mesh. Try to get a single Trimesh object.
        loaded_mesh_data: Union[trimesh.Trimesh, trimesh.Scene, Any] = trimesh.load(original_mesh_path, process=True)
        
        mesh: Optional[trimesh.Trimesh] = None
        if isinstance(loaded_mesh_data, trimesh.Trimesh):
            mesh = loaded_mesh_data
        elif isinstance(loaded_mesh_data, trimesh.Scene):
            mesh_list = [geom for geom in loaded_mesh_data.geometry.values() if isinstance(geom, trimesh.Trimesh)]
            if not mesh_list:
                print(f"Warning: No trimesh geometries found in scene: {original_mesh_path}")
                return None, None
            if len(mesh_list) == 1:
                mesh = mesh_list[0]
            else:
                print(f"Info: Scene contains multiple geometries. Concatenating them.")
                mesh = trimesh.util.concatenate(mesh_list)
        else:
            print(f"Warning: Loaded object is not a Trimesh or Scene: {type(loaded_mesh_data)}")
            return None, None

        if not mesh or not hasattr(mesh, 'faces'):
             print(f"Error: Could not load a valid Trimesh object from: {original_mesh_path}")
             return None, None

        if not os.path.exists(segmentation_npy_path):
            print(f"Error: Segmentation npy file not found: {segmentation_npy_path}")
            return None, None
        
        seg_map = np.load(segmentation_npy_path)

        if len(seg_map) != len(mesh.faces):
            print(f"Warning: Segmentation map length ({len(seg_map)}) does not match mesh face count ({len(mesh.faces)}).")
            if len(seg_map) == len(mesh.vertices):
                print("Info: Attempting to map vertex segmentation to faces by first vertex's segment ID.")
                face_segment_ids = np.zeros(len(mesh.faces), dtype=int)
                for i, face in enumerate(mesh.faces):
                    vertex_segments = seg_map[face]
                    face_segment_ids[i] = vertex_segments[0]
                seg_map = face_segment_ids
                
                if len(seg_map) != len(mesh.faces):
                     print(f"Error: Vertex-to-face segmentation mapping failed. Final seg_map length: {len(seg_map)} vs faces {len(mesh.faces)}")
                     return None, None
                print(f"Info: Successfully mapped vertex segmentation to {len(seg_map)} faces.")
            else:
                print(f"Error: Segmentation map length ({len(seg_map)}) does not match face count ({len(mesh.faces)}) or vertex count ({len(mesh.vertices)}). Cannot proceed.")
                return None, None
        
        os.makedirs(output_dir, exist_ok=True)

        all_target_face_indices = []
        unique_target_ids = sorted(list(set(target_segment_ids))) 

        for target_id in unique_target_ids:
            indices = np.where(seg_map == target_id)[0]
            all_target_face_indices.extend(indices)
        
        all_target_face_indices = np.unique(all_target_face_indices).astype(int)
        
        max_face_idx = len(mesh.faces) - 1
        valid_target_face_indices = all_target_face_indices[all_target_face_indices <= max_face_idx]

        all_face_indices = np.arange(len(mesh.faces))
        remaining_face_indices = np.setdiff1d(all_face_indices, valid_target_face_indices)

        target_submesh_path_final = None
        remaining_submesh_path_final = None

        ids_str = "_".join(map(str, unique_target_ids))
        
        if len(valid_target_face_indices) > 0:
            target_submesh = mesh.submesh([valid_target_face_indices], append=False)
            if isinstance(target_submesh, trimesh.Trimesh) and len(target_submesh.faces) > 0:
                target_submesh_filename = f"{base_output_name}_target_{ids_str}.obj"
                target_submesh_path_final = os.path.join(output_dir, target_submesh_filename)
                target_submesh.export(target_submesh_path_final)
                print(f"Target submesh saved to: {target_submesh_path_final} with {len(target_submesh.faces)} faces.")
            else:
                print(f"Warning: Target submesh (IDs: {ids_str}) is empty or invalid after submeshing.")
        else:
            print(f"Info: No faces found for target segment IDs: {unique_target_ids}. Target submesh will not be created.")

        if len(remaining_face_indices) > 0:
            remaining_submesh = mesh.submesh([remaining_face_indices], append=False)
            if isinstance(remaining_submesh, trimesh.Trimesh) and len(remaining_submesh.faces) > 0:
                remaining_submesh_filename = f"{base_output_name}_remaining_{ids_str}.obj"
                remaining_submesh_path_final = os.path.join(output_dir, remaining_submesh_filename)
                remaining_submesh.export(remaining_submesh_path_final)
                print(f"Remaining submesh saved to: {remaining_submesh_path_final} with {len(remaining_submesh.faces)} faces.")
            else:
                print(f"Warning: Remaining submesh (related to target IDs: {ids_str}) is empty or invalid after submeshing.")
        else:
            print(f"Info: No remaining faces after extracting target segments (IDs: {unique_target_ids}). Remaining submesh will not be created.")
            
        return target_submesh_path_final, remaining_submesh_path_final

    except Exception as e:
        print(f"An error occurred during mesh splitting: {e}")
        import traceback
        traceback.print_exc()
        return None, None
