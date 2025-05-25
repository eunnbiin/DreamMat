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
    """
    Splits a mesh into target and remaining submeshes based on segmentation information.
    Saves submeshes as .obj files.
    Includes detailed debugging prints.
    """
    print(f"\n--- [mesh_utils.py DEBUG] ---")
    print(f"Function split_mesh_by_segmentation called with:")
    print(f"  original_mesh_path: {original_mesh_path}")
    print(f"  segmentation_npy_path: {segmentation_npy_path}")
    print(f"  target_segment_ids: {target_segment_ids}")
    print(f"  output_dir: {output_dir}")
    print(f"  base_output_name: {base_output_name}")
    print(f"--- [mesh_utils.py DEBUG] ---\n")

    try:
        if not os.path.exists(original_mesh_path):
            print(f"[DEBUG mesh_utils] Error: Original mesh file not found: {original_mesh_path}")
            return None, None

        print(f"[DEBUG mesh_utils] Loading mesh from: {original_mesh_path} (Attempting sep.py style loading)")
        # process=False로 변경하고, sep.py와 유사한 Scene 처리 로직 적용
        _loaded_mesh_data_raw = trimesh.load(original_mesh_path, process=False) 
        
        mesh: Optional[trimesh.Trimesh] = None
        if isinstance(_loaded_mesh_data_raw, trimesh.Scene):
            print(f"[DEBUG mesh_utils] Loaded as Scene. Applying .to_geometry()")
            try:
                mesh = _loaded_mesh_data_raw.to_geometry()
                # to_geometry()가 간혹 Scene을 반환하는 경우가 있어서 한 번 더 체크
                if isinstance(mesh, trimesh.Scene): # 여전히 Scene이면 concatenate 시도
                     print(f"[DEBUG mesh_utils] .to_geometry() still returned a Scene. Concatenating components.")
                     mesh = trimesh.util.concatenate([g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)])
            except Exception as e_to_geom:
                print(f"[DEBUG mesh_utils] Error during .to_geometry(): {e_to_geom}. Falling back to concatenate.")
                # Scene의 geometry들을 직접 concatenate
                mesh_list_from_scene = [geom for geom in _loaded_mesh_data_raw.geometry.values() if isinstance(geom, trimesh.Trimesh)]
                if mesh_list_from_scene:
                    mesh = trimesh.util.concatenate(mesh_list_from_scene)
                else:
                    print(f"[DEBUG mesh_utils] No trimesh geometries in scene for fallback concatenate.")
        elif isinstance(_loaded_mesh_data_raw, trimesh.Trimesh):
            print(f"[DEBUG mesh_utils] Loaded directly as Trimesh.")
            mesh = _loaded_mesh_data_raw
        else: # Scene도 Trimesh도 아닌 경우 (예: list of Trimesh 등)
            print(f"[DEBUG mesh_utils] Loaded as {type(_loaded_mesh_data_raw)}. Attempting concatenate if it's a list.")
            try:
                # sep.py의 elif not isinstance(mesh, trimesh.Trimesh): 부분과 유사하게 처리
                if isinstance(_loaded_mesh_data_raw, list) and all(isinstance(m, trimesh.Trimesh) for m in _loaded_mesh_data_raw):
                     mesh = trimesh.util.concatenate(_loaded_mesh_data_raw)
                else: # 그 외에는 명시적으로 Trimesh가 아닌 경우 에러보다는 경고 후 None 처리
                     print(f"[DEBUG mesh_utils] Could not convert to single Trimesh. Type was {type(_loaded_mesh_data_raw)}")
            except Exception as e_concat_other:
                print(f"[DEBUG mesh_utils] Warning: Could not convert loaded object of type {type(_loaded_mesh_data_raw)} to single Trimesh: {e_concat_other}")

        if mesh and isinstance(mesh, trimesh.Trimesh):
            print(f"[DEBUG mesh_utils] Final mesh (sep.py style). Faces: {len(mesh.faces)}, Vertices: {len(mesh.vertices)}")
        else:
            print(f"[DEBUG mesh_utils] Failed to obtain a valid Trimesh object using sep.py style loading.")
            return None, None

        if not mesh or not hasattr(mesh, 'faces') or len(mesh.faces) == 0:
             print(f"[DEBUG mesh_utils] Error: Could not load a valid Trimesh object with faces from: {original_mesh_path}")
             return None, None

        if not os.path.exists(segmentation_npy_path):
            print(f"[DEBUG mesh_utils] Error: Segmentation npy file not found: {segmentation_npy_path}")
            return None, None
        
        print(f"[DEBUG mesh_utils] Loading segmentation map from: {segmentation_npy_path}")
        seg_map_original = np.load(segmentation_npy_path)
        seg_map = seg_map_original.copy() # 원본 유지를 위해 복사본 사용
        print(f"[DEBUG mesh_utils] Loaded segmentation map. Shape: {seg_map.shape}, Unique IDs: {np.unique(seg_map)}")
        print(f"[DEBUG mesh_utils] Mesh face count: {len(mesh.faces)}, Vertex count: {len(mesh.vertices)}")


        if len(seg_map) != len(mesh.faces):
            print(f"[DEBUG mesh_utils] Warning: Segmentation map length ({len(seg_map)}) does not match mesh face count ({len(mesh.faces)}).")
            if len(seg_map) == len(mesh.vertices):
                print("[DEBUG mesh_utils] Info: Segmentation map length matches vertex count. Attempting to map vertex segmentation to faces.")
                face_segment_ids = np.zeros(len(mesh.faces), dtype=int)
                valid_face_map_count = 0
                for i, face_verts in enumerate(mesh.faces):
                    # Ensure face_verts indices are within bounds of seg_map (vertex based)
                    if np.all(face_verts < len(seg_map)):
                        vertex_segments_for_face = seg_map[face_verts]
                        # Using the segment ID of the first vertex for the face
                        face_segment_ids[i] = vertex_segments_for_face[0]
                        valid_face_map_count +=1
                    else:
                        print(f"[DEBUG mesh_utils] Error: Face {i} has vertex indices {face_verts} out of bounds for seg_map of length {len(seg_map)}")
                        face_segment_ids[i] = -99 # Invalid mapping marker
                
                seg_map = face_segment_ids
                print(f"[DEBUG mesh_utils] Vertex-to-face mapping: {valid_face_map_count} faces mapped. New seg_map unique IDs: {np.unique(seg_map)}")
                
                if len(seg_map) != len(mesh.faces): # 최종 확인
                     print(f"[DEBUG mesh_utils] Error: Vertex-to-face segmentation mapping failed to produce map of correct length. Final seg_map length: {len(seg_map)}")
                     return None, None
            else:
                print(f"[DEBUG mesh_utils] Error: Segmentation map length ({len(seg_map)}) does not match face count ({len(mesh.faces)}) or vertex count ({len(mesh.vertices)}). Cannot proceed.")
                return None, None
        else:
            print(f"[DEBUG mesh_utils] Info: Segmentation map length matches face count. Using as is.")
        
        os.makedirs(output_dir, exist_ok=True)
        print(f"[DEBUG mesh_utils] Ensured output directory exists: {output_dir}")

        all_target_face_indices = []
        unique_target_ids = sorted(list(set(target_segment_ids))) 
        print(f"[DEBUG mesh_utils] Unique target_segment_ids to find: {unique_target_ids}")

        for target_id in unique_target_ids:
            # 현재 seg_map (면 기준일 것임)에서 target_id를 가진 면들의 인덱스를 찾음
            indices = np.where(seg_map == target_id)[0]
            if len(indices) > 0:
                print(f"[DEBUG mesh_utils] Found {len(indices)} faces for target_id {target_id}.")
            else:
                print(f"[DEBUG mesh_utils] Found 0 faces for target_id {target_id}.")
            all_target_face_indices.extend(indices)
        
        if not all_target_face_indices:
            print(f"[DEBUG mesh_utils] No faces found for any of the target_segment_ids. Target submesh cannot be created.")
        else:
            all_target_face_indices = np.unique(all_target_face_indices).astype(int)
            print(f"[DEBUG mesh_utils] Total unique target face indices found: {len(all_target_face_indices)}. Indices: {all_target_face_indices[:20]}...") # 처음 20개만 출력


        max_face_idx = len(mesh.faces) - 1
        # 유효한 인덱스 필터링 (너무 크거나 음수인 인덱스 제거)
        valid_target_face_indices = all_target_face_indices[(all_target_face_indices >= 0) & (all_target_face_indices <= max_face_idx)]
        
        if len(valid_target_face_indices) != len(all_target_face_indices):
            print(f"[DEBUG mesh_utils] Filtered out invalid face indices. Original count: {len(all_target_face_indices)}, Valid count: {len(valid_target_face_indices)}")
        
        all_mesh_face_indices = np.arange(len(mesh.faces))
        remaining_face_indices = np.setdiff1d(all_mesh_face_indices, valid_target_face_indices)
        print(f"[DEBUG mesh_utils] Remaining face indices count: {len(remaining_face_indices)}")


        target_submesh_path_final = None
        remaining_submesh_path_final = None

        ids_str = "_".join(map(str, unique_target_ids)) # 파일명용 ID 문자열
        
        if len(valid_target_face_indices) > 0:
            print(f"[DEBUG mesh_utils] Attempting to create target submesh with {len(valid_target_face_indices)} faces using append=True.")
            # append=True로 변경하여 시도
            potential_target_submesh = mesh.submesh([valid_target_face_indices], append=True) 
            
            actual_target_submesh = None
            if isinstance(potential_target_submesh, trimesh.Trimesh):
                actual_target_submesh = potential_target_submesh
                print(f"[DEBUG mesh_utils] trimesh.submesh returned a Trimesh object directly for target.")
            elif isinstance(potential_target_submesh, list):
                print(f"[DEBUG mesh_utils] trimesh.submesh returned a list for target. Length: {len(potential_target_submesh)}")
                if len(potential_target_submesh) == 1 and isinstance(potential_target_submesh[0], trimesh.Trimesh):
                    actual_target_submesh = potential_target_submesh[0]
                    print(f"[DEBUG mesh_utils] Using the first Trimesh object from the list for target.")
                elif len(potential_target_submesh) > 1:
                    print(f"[DEBUG mesh_utils] Concatenating multiple Trimesh objects from the list for target.")
                    try:
                        actual_target_submesh = trimesh.util.concatenate(
                            [m for m in potential_target_submesh if isinstance(m, trimesh.Trimesh)]
                        )
                    except Exception as e:
                        print(f"[DEBUG mesh_utils] Error concatenating target submeshes: {e}")
                else:
                    print(f"[DEBUG mesh_utils] List from trimesh.submesh for target is empty or contains non-Trimesh objects.")
            
            if actual_target_submesh and isinstance(actual_target_submesh, trimesh.Trimesh) and len(actual_target_submesh.faces) > 0:
                target_submesh_filename = f"{base_output_name}_target_{ids_str}.glb" # 변경 .obj -> .glb
                target_submesh_path_final = os.path.join(output_dir, target_submesh_filename)
                actual_target_submesh.export(target_submesh_path_final) # trimesh는 파일 확장자로 포맷 자동 감지
                print(f"[DEBUG mesh_utils] Target submesh successfully created and saved to: {target_submesh_path_final} with {len(actual_target_submesh.faces)} faces.")
            else:
                print(f"[DEBUG mesh_utils] Warning: Target submesh (IDs: {ids_str}) is empty or invalid. Processed type: {type(actual_target_submesh)}")
        else:
            print(f"[DEBUG mesh_utils] Info: No valid faces found for target segment IDs: {unique_target_ids}. Target submesh will not be created.")

        if len(remaining_face_indices) > 0:
            print(f"[DEBUG mesh_utils] Attempting to create remaining submesh with {len(remaining_face_indices)} faces using append=True.")
            # append=True로 변경하여 시도
            potential_remaining_submesh = mesh.submesh([remaining_face_indices], append=True)
            
            actual_remaining_submesh = None
            if isinstance(potential_remaining_submesh, trimesh.Trimesh):
                actual_remaining_submesh = potential_remaining_submesh
                print(f"[DEBUG mesh_utils] trimesh.submesh returned a Trimesh object directly for remaining.")
            elif isinstance(potential_remaining_submesh, list):
                print(f"[DEBUG mesh_utils] trimesh.submesh returned a list for remaining. Length: {len(potential_remaining_submesh)}")
                if len(potential_remaining_submesh) == 1 and isinstance(potential_remaining_submesh[0], trimesh.Trimesh):
                    actual_remaining_submesh = potential_remaining_submesh[0]
                    print(f"[DEBUG mesh_utils] Using the first Trimesh object from the list for remaining.")
                elif len(potential_remaining_submesh) > 1:
                    print(f"[DEBUG mesh_utils] Concatenating multiple Trimesh objects from the list for remaining.")
                    try:
                        actual_remaining_submesh = trimesh.util.concatenate(
                             [m for m in potential_remaining_submesh if isinstance(m, trimesh.Trimesh)]
                        )
                    except Exception as e:
                         print(f"[DEBUG mesh_utils] Error concatenating remaining submeshes: {e}")
                else:
                    print(f"[DEBUG mesh_utils] List from trimesh.submesh for remaining is empty or contains non-Trimesh objects.")

            if actual_remaining_submesh and isinstance(actual_remaining_submesh, trimesh.Trimesh) and len(actual_remaining_submesh.faces) > 0:
                remaining_submesh_filename = f"{base_output_name}_remaining_{ids_str}.glb" # 변경 .obj -> .glb
                remaining_submesh_path_final = os.path.join(output_dir, remaining_submesh_filename)
                actual_remaining_submesh.export(remaining_submesh_path_final) # trimesh는 파일 확장자로 포맷 자동 감지
                print(f"[DEBUG mesh_utils] Remaining submesh successfully created and saved to: {remaining_submesh_path_final} with {len(actual_remaining_submesh.faces)} faces.")
            else:
                print(f"[DEBUG mesh_utils] Warning: Remaining submesh (related to target IDs: {ids_str}) is empty or invalid. Processed type: {type(actual_remaining_submesh)}")
        else:
            print(f"[DEBUG mesh_utils] Info: No remaining faces after extracting target segments (IDs: {unique_target_ids}). Remaining submesh will not be created.")
        
        if not target_submesh_path_final and not remaining_submesh_path_final:
             print(f"[DEBUG mesh_utils] CRITICAL: Neither target nor remaining submesh was created.")

        return target_submesh_path_final, remaining_submesh_path_final

    except Exception as e:
        print(f"[DEBUG mesh_utils] An error occurred during mesh splitting: {e}")
        import traceback
        traceback.print_exc()
        return None, None
