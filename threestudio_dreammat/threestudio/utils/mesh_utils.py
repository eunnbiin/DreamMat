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

        _loaded_mesh_data_raw = trimesh.load(original_mesh_path, process=False, maintain_order=True, force='mesh')
        
        mesh: Optional[trimesh.Trimesh] = None
        if isinstance(_loaded_mesh_data_raw, trimesh.Trimesh):
            mesh = _loaded_mesh_data_raw
        elif isinstance(_loaded_mesh_data_raw, trimesh.Scene):
            mesh_list_from_scene = [geom for geom in _loaded_mesh_data_raw.geometry.values() if isinstance(geom, trimesh.Trimesh)]
            if mesh_list_from_scene:
                try:
                    mesh = trimesh.util.concatenate(mesh_list_from_scene)
                except Exception as e_concat:
                    print(f"Error during scene concatenation: {e_concat}")
                    return None, None
            else:
                print(f"No trimesh geometries found in scene for concatenation.")
                return None, None
        else:
             print(f"Loaded object type {type(_loaded_mesh_data_raw)} is not Trimesh or Scene. Cannot process.")
             return None, None

        if not mesh or not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
             print(f"Error: Could not load a valid Trimesh object with faces from: {original_mesh_path}")
             return None, None

        if not os.path.exists(segmentation_npy_path):
            print(f"Error: Segmentation npy file not found: {segmentation_npy_path}")
            return None, None
        
        seg_map = np.load(segmentation_npy_path)

        if len(seg_map) != len(mesh.faces):
            if len(seg_map) == len(mesh.vertices):
                face_segment_ids = np.zeros(len(mesh.faces), dtype=int)
                for i, face_verts_indices in enumerate(mesh.faces):
                    if np.all(face_verts_indices < len(seg_map)):
                        face_segment_ids[i] = seg_map[face_verts_indices[0]]
                    else:
                        print(f"Error: Face {i} has vertex index out of bounds for vertex-based seg_map. Face indices: {face_verts_indices}, mat length: {len(seg_map)}")
                        return None, None
                seg_map = face_segment_ids
                if len(seg_map) != len(mesh.faces):
                     print(f"Error: Vertex-to-face segmentation mapping failed to produce map of correct length.")
                     return None, None
            else:
                print(f"Error: Segmentation map length ({len(seg_map)}) does not match face count ({len(mesh.faces)}) or vertex count ({len(mesh.vertices)}). Cannot proceed.")
                return None, None
        
        os.makedirs(output_dir, exist_ok=True)

        all_target_face_indices = []
        unique_target_ids = sorted(list(set(target_segment_ids))) 

        for target_id in unique_target_ids:
            indices = np.where(seg_map == target_id)[0]
            all_target_face_indices.extend(indices)
        
        if all_target_face_indices:
            all_target_face_indices = np.unique(all_target_face_indices).astype(int)

        max_face_idx = len(mesh.faces) - 1
        valid_target_face_indices = all_target_face_indices[(all_target_face_indices >= 0) & (all_target_face_indices <= max_face_idx)]
        
        all_mesh_face_indices = np.arange(len(mesh.faces))
        remaining_face_indices = np.setdiff1d(all_mesh_face_indices, valid_target_face_indices)

        target_submesh_path_final = None
        remaining_submesh_path_final = None

        ids_str = "_".join(map(str, unique_target_ids))
        
        if len(valid_target_face_indices) > 0:
            potential_target_submesh = mesh.submesh([valid_target_face_indices], append=True) 
            actual_target_submesh = None
            if isinstance(potential_target_submesh, trimesh.Trimesh):
                actual_target_submesh = potential_target_submesh
            elif isinstance(potential_target_submesh, list):
                if len(potential_target_submesh) == 1 and isinstance(potential_target_submesh[0], trimesh.Trimesh):
                    actual_target_submesh = potential_target_submesh[0]
                elif len(potential_target_submesh) > 1:
                    try:
                        actual_target_submesh = trimesh.util.concatenate(
                            [m for m in potential_target_submesh if isinstance(m, trimesh.Trimesh) and len(m.faces)>0]
                        )
                    except Exception as e_concat_target:
                        print(f"Error concatenating target submeshes: {e_concat_target}")
            if actual_target_submesh and isinstance(actual_target_submesh, trimesh.Trimesh) and len(actual_target_submesh.faces) > 0:
                target_submesh_filename = f"{base_output_name}_target_{ids_str}.glb"
                target_submesh_path_final = os.path.join(output_dir, target_submesh_filename)
                actual_target_submesh.export(target_submesh_path_final)
                print(f"Target submesh saved to: {target_submesh_path_final}")
            else:
                print(f"Warning: Target submesh (IDs: {ids_str}) could not be created or is empty.")

        if len(remaining_face_indices) > 0:
            potential_remaining_submesh = mesh.submesh([remaining_face_indices], append=True)
            actual_remaining_submesh = None
            if isinstance(potential_remaining_submesh, trimesh.Trimesh):
                actual_remaining_submesh = potential_remaining_submesh
            elif isinstance(potential_remaining_submesh, list):
                if len(potential_remaining_submesh) == 1 and isinstance(potential_remaining_submesh[0], trimesh.Trimesh):
                    actual_remaining_submesh = potential_remaining_submesh[0]
                elif len(potential_remaining_submesh) > 1:
                    try:
                        actual_remaining_submesh = trimesh.util.concatenate(
                             [m for m in potential_remaining_submesh if isinstance(m, trimesh.Trimesh) and len(m.faces)>0]
                        )
                    except Exception as e_concat_remaining:
                         print(f"Error concatenating remaining submeshes: {e_concat_remaining}")
            if actual_remaining_submesh and isinstance(actual_remaining_submesh, trimesh.Trimesh) and len(actual_remaining_submesh.faces) > 0:
                remaining_submesh_filename = f"{base_output_name}_remaining_{ids_str}.glb"
                remaining_submesh_path_final = os.path.join(output_dir, remaining_submesh_filename)
                actual_remaining_submesh.export(remaining_submesh_path_final)
                print(f"Remaining submesh saved to: {remaining_submesh_path_final}")
            else:
                print(f"Warning: Remaining submesh (related to target IDs: {ids_str}) could not be created or is empty.")
        
        return target_submesh_path_final, remaining_submesh_path_final

    except Exception as e:
        print(f"An error occurred during mesh splitting in mesh_utils.py: {e}")
        import traceback
        traceback.print_exc()
        return None, None
