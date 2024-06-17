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

def process_file(filename: str) -> np.ndarray:
    bpy.ops.import_scene.fbx(filepath=filename)
    frame_start = bpy.context.scene.frame_start
    frame_end = bpy.context.scene.frame_end
    print(f'frame_start: {frame_start}, frame_end: {frame_end}')
    sequence = []
    for frame_number in range(frame_start, frame_end + 1):
        print(f'Processing frame {frame_number}')
        bpy.context.scene.frame_set(frame_number)
        armature_name = "Root"
        armature = bpy.data.objects.get(armature_name)
        locations = []
        if armature is None:
            print(f"Armature '{armature_name}' not found.")
        else:
            locations = compute_bone_world_space_coordinates(armature)
        sequence.append(locations)
    return np.array(sequence)