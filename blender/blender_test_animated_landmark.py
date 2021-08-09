import bpy
import sys
import os

dir = "/home/alkhemi/Documents/thesis/speech-animation-training-tools/blender/"
if not dir in sys.path:
    sys.path.append(dir)

from blender_DisplacementLandmarkPredictor import DisplacementLandmarkPredictor
import pandas as pd


# video_filepath = "/mnt/data/thesis data jangan dihapus/video/s19/bbby1s.mpg"
wav_filepath = "/mnt/data/thesis data jangan dihapus/audio/s32/pwwu1s.wav"
video_filepath = "/mnt/data/thesis data jangan dihapus/video/s32/pwwu1s.mpg"
# wav_filepath = "/home/alkhemi/untitled.wav"
identity_landmark = "/home/alkhemi/Documents/thesis/animated/s19/bbby1s_anim.csv"

scene = bpy.context.scene 

if not scene.sequence_editor:
    scene.sequence_editor_create()

for seq in scene.sequence_editor.sequences:
    scene.sequence_editor.sequences.remove(seq)

#Sequences.new_sound(name, filepath, channel, frame_start)    
soundstrip = scene.sequence_editor.sequences.new_sound("speech", video_filepath, 2, 0)
videostrip = scene.sequence_editor.sequences.new_movie("video", video_filepath, 1, 0)


predictor = DisplacementLandmarkPredictor(
    X_feature_size=39,
    Y_feature_size=107,
    x_num_pre=12,
    x_num_post=12,
    y_num_pre=0,
    y_num_post=0,
    d_model=64,
    num_heads=8,
    dff=256,
    model_weights=dir+"Attention64_h8_dff256_batch64_lr1e-06-100-0.0036.h5",
    x_scaler=dir+"x_scaler_coef.csv",
    y_scaler=dir+"y_scaler_coef.csv",
)

animation_magnitude = 2

identity_landmark_df = pd.read_csv(identity_landmark).iloc[0]
displacement_landmark_df = predictor.predictAsDF(wav_filepath)

effectstrip = scene.sequence_editor.sequences.new_effect("speed","SPEED", channel=3, frame_start=0,frame_end=len(displacement_landmark_df),seq1=videostrip)
scene.sequence_editor.sequences_all["video"].frame_final_duration = len(displacement_landmark_df)

collection = bpy.data.collections.get("FaceLandmarkAnimation")
if collection:
    for o in collection.children:
        bpy.data.objects.remove(o)
    bpy.context.scene.collection.children.unlink(collection)
    bpy.data.collections.remove(collection)


collection = bpy.data.collections.new("FaceLandmarkAnimation")
bpy.context.scene.collection.children.link(collection)

# NOTE the use of 'collection.name' to account for potential automatic renaming
layer_collection = bpy.context.view_layer.layer_collection.children[collection.name]
bpy.context.view_layer.active_layer_collection = layer_collection


for i in range(68):
    if bpy.data.objects.get(f"LM_{i}"):
        bpy.data.objects.remove(bpy.data.objects.get(f"LM_{i}"))
    if bpy.data.meshes.get(f"LM_{i}"):
        bpy.data.meshes.remove(bpy.data.meshes.get(f"LM_{i}"))

    bpy.ops.mesh.primitive_uv_sphere_add(
        radius=0.001,
        enter_editmode=False,
        align="WORLD",
        location=(
            identity_landmark_df[f"X_{i}"] / 1000,
            identity_landmark_df[f"Z_{i}"] / 1000,
            identity_landmark_df[f"Y_{i}"] / 1000,
        ),
    )
    ob = bpy.context.object
    me = ob.data
    ob.name = f"LM_{i}"
    me.name = f"LM_{i}"


# Simple but ULTRA MEGA SLOW
# for frame in range(len(displacement_landmark_df)):
#     for i in range(68):
# bpy.context.scene.frame_set(frame)
# ob = bpy.data.objects[f"LM_{i}"]
# ob.location = (
#     (identity_landmark_df[f"X_{i}"]+displacement_landmark_df[f"X_{i}"][frame])/1000,
#     (identity_landmark_df[f"Z_{i}"]+displacement_landmark_df[f"Z_{i}"][frame])/1000,
#     (identity_landmark_df[f"Y_{i}"]+displacement_landmark_df[f"Y_{i}"][frame])/1000,
# )
# ob.keyframe_insert(data_path="location",index=-1)

# FAST
for i in range(68):
    if bpy.data.actions.get(f"LM_{i}_Action"):
        bpy.data.actions.remove(bpy.data.actions.get(f"LM_{i}_Action"))

    obj = bpy.data.objects[f"LM_{i}"]
    obj.animation_data_create()
    obj.animation_data.action = bpy.data.actions.new(name=f"LM_{i}_Action")

    fcu_z = obj.animation_data.action.fcurves.new(data_path="location", index=0)
    fcu_z.keyframe_points.add(len(displacement_landmark_df))
    for frame in range(len(displacement_landmark_df)):
        fcu_z.keyframe_points[frame].co = (
            frame,
            (
                identity_landmark_df[f"X_{i}"]
                + animation_magnitude * displacement_landmark_df[f"X_{i}"][frame]
            )
            / 1000,
        )
    fcu_z = obj.animation_data.action.fcurves.new(data_path="location", index=1)
    fcu_z.keyframe_points.add(len(displacement_landmark_df))
    for frame in range(len(displacement_landmark_df)):
        fcu_z.keyframe_points[frame].co = (
            frame,
            (
                identity_landmark_df[f"Z_{i}"]
                + animation_magnitude * displacement_landmark_df[f"Z_{i}"][frame]
            )
            / 1000,
        )
    fcu_z = obj.animation_data.action.fcurves.new(data_path="location", index=2)
    fcu_z.keyframe_points.add(len(displacement_landmark_df))
    for frame in range(len(displacement_landmark_df)):
        fcu_z.keyframe_points[frame].co = (
            frame,
            (
                identity_landmark_df[f"Y_{i}"]
                + animation_magnitude * displacement_landmark_df[f"Y_{i}"][frame]
            )
            / 1000,
        )
