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