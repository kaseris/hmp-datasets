import os
import os.path as osp
import pickle
import argparse

import bpy
import numpy as np

def compute_bone_world_space_coordinates(armature):
    locations = []
    for bone in armature.pose.bones:
        bone_name = bone.name
        bone_location = (
            armature.matrix_world @ bone.matrix
        ) @ (bone.head - bone.tail)  # Transform to world space
        locations.append(np.array(bone_location))
    return np.array(locations)


def get_animation_range(armature_name):
    if armature_name not in bpy.data.objects:
        print(f'Armature `{armature_name} not found in the scene.`')
        return None, None
    armature = bpy.data.objects[armature_name]

    if armature.type != 'ARMATURE':
        print(f'Object `{armature_name}` is not an armature')
        return None, None
    
    start_frame = float('inf')
    end_frame = float('-inf')

    for fcurve in armature.animation_data.action.fcurves:
        for keyframe in fcurve.keyframe_points:
            frame = keyframe.co.x
            start_frame = min(start_frame, frame)
            end_frame = max(end_frame, frame)

    return int(start_frame), int(end_frame)

def get_bone_hierarchy(armature):
    def recurse(bone, parent=None):
        hierarchy = {
            'name': bone.name,
            'parent': parent,
            'children': [recurse(child, bone.name) for child in bone.children]
        }
        return hierarchy
    
    return [recurse(bone) for bone in armature.data.bones if bone.parent is None]

def flatten_hierarchy(hierarchy):
    flat = []
    for bone in hierarchy:
        flat.append((bone['name'], bone['parent']))
        flat.extend(flatten_hierarchy(bone['children']))
    return flat

def generate_motion_paths(armature_name, start_frame, end_frame):
    if armature_name not in bpy.data.objects:
        print(f"Armature '{armature_name}' not found in the scene.")
        return None

    armature = bpy.data.objects[armature_name]
    
    if armature.type != 'ARMATURE':
        print(f"Object '{armature_name}' is not an armature.")
        return None

    motion_paths = {}
    bone_hierarchy = get_bone_hierarchy(armature)
    flat_hierarchy = flatten_hierarchy(bone_hierarchy)

    current_frame = bpy.context.scene.frame_current

    for frame in range(start_frame, end_frame + 1):
        bpy.context.scene.frame_set(frame)
        
        for bone in armature.pose.bones:
            if bone.name not in motion_paths:
                motion_paths[bone.name] = []
            
            world_pos = armature.matrix_world @ bone.matrix @ bone.head
            motion_paths[bone.name].append(world_pos)

    for bone_name in motion_paths:
        motion_paths[bone_name] = np.array(motion_paths[bone_name])

    bpy.context.scene.frame_set(current_frame)

    return motion_paths, flat_hierarchy


def process_file(filename: str) -> np.ndarray:
    bpy.ops.import_scene.fbx(filepath=filename)
    frame_start = bpy.context.scene.frame_start
    frame_end = bpy.context.scene.frame_end
    if args.verbose:
        print(f'frame_start: {frame_start}, frame_end: {frame_end}')
    sequence = []
    for frame_number in range(frame_start, frame_end + 1):
        if args.verbose:
            print(f'Processing frame {frame_number}')
        bpy.context.scene.frame_set(frame_number)
        armature_name = "Root"
        armature = bpy.data.objects.get(armature_name)
        locations = []
        if armature is None:
            raise ValueError(f"Armature '{armature_name}' not found.")
        else:
            locations = compute_bone_world_space_coordinates(armature)
        sequence.append(locations)
    return np.array(sequence)

def process_file_with_motion_paths(fbx_path: str):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.fbx(filepath=fbx_path)
    start, end = get_animation_range('Root')
    if start is not None and end is not None:
        print(f"Animation range for 'Root': Start frame = {start}, End frame = {end}")
        motion_paths, hierarchy = generate_motion_paths('Root', start, end)
        if motion_paths:
            print(f'Motion paths were read correctly.')
            print("\nMotion Path Data:")
            for bone_name, path in motion_paths.items():
                print(f"\nBone: {bone_name}")
                print(f"Shape of path array: {path.shape}")
                print(f"First point: {path[0]}")
                print(f"Last point: {path[-1]}")
                
                # Example of numpy operations
                mean_position = np.mean(path, axis=0)
                print(f"Mean position: {mean_position}")
                
                # Calculate total distance traveled
                distances = np.linalg.norm(path[1:] - path[:-1], axis=1)
                total_distance = np.sum(distances)
                print(f"Total distance traveled: {total_distance}")
        else:
            print('No motion paths were generated.')
    else:
        print('Could not determine animation range.')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', help='Raw data directory', type=str, required=True)
    parser.add_argument('--verbose', help='Enable print statements', type=bool, required=False, default=False)
    global args
    args = parser.parse_args()
    subdirs = os.listdir(args.data_dir)
    if 'recordings' not in subdirs:
        raise ValueError('`recordings` subdirectory not found.')
    raw_data_path = osp.join(args.data_dir, 'recordings')
    fnames = os.listdir(raw_data_path)
    sequences_dict = dict()
    for fname in fnames:
        if args.verbose:
            print(f'Processing file: {fname}')
        full_path = osp.join(raw_data_path, fname)
        sequence = process_file(full_path)
        sequences_dict[f'{fname}'] = sequence
    with open('rokoko_data.pkl', 'wb') as f:
        pickle.dump(sequences_dict, f)