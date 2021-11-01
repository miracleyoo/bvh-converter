import bpy
import os.path as op
import numpy as np
import pickle as pkl
C = bpy.context
D = bpy.data

#####################################################################
################# PARAMETERS YOU NEED TO FILL IN ####################
model_name='Tda Hood Miku 1.00 by iRon0129'
output_dir = r'D:\MMD\Setting_Pairs\miku_neutral_street_bluestar'
output_flags = {'raw': True, 'kp13': False, 'kp13_normed': True}
start_custom = 6  # Blender will add 5 frames at beginning than MMD.
end_custom = None # If you want only part of the frames, set it.
norm_level = 2 # Select from (0,1,2).
# 0: Only centering the model xy coordinates.
# 1: Based on 1, model's lowest z axis coordinate is pulled to 0.
# 2: Based on 2, the highest z axis coordinate is normalized to 1.
#####################################################################

# Get part of the data using bone names
def pose_slice(names, data):
    if type(names) not in (list, tuple):
        names = [str(names)]
    indexes = np.array([data['names'].index(k) for k in names])
    joint_arr = data['data'][:,indexes]
    return joint_arr

# The armature of a imported model is different from the model object
# itself. The armature of a certain model contains the pose bones, as
# well as the position of bone head and tail, rotation, and so on.
# Usually, its name is the original model object name + '_arm'. 
armature_name = model_name + '_arm'
frame_start = C.scene.frame_start if start_custom is None else start_custom
frame_end = C.scene.frame_end if end_custom is None else end_custom

# These are the Japanese names of the 13 keypoints of a human.
parts_names_kp13 = ['頭', '腕.R', '腕.L', 'ひじ.R', 'ひじ.L', '足.R', '足.L', 
        '手首.R', '手首.L',  'ひざ.R', 'ひざ.L', '足首.R', '足首.L']
# Use the center point 'グルーブ' as the center x and y coordinate.
center_name = 'グルーブ'

# Get the POSE bones objects.
pose_bones = {pb.name: pb for pb in D.objects[armature_name].pose.bones}
pb_names = list(pose_bones.keys())

# Iterate over all the valid frames, and keep track of all the bones
# in your selected pose model. Save them to `data` numpy array. 
if not output_flags['raw']:
    pb_names = parts_names_kp13
joint_arr = np.zeros((frame_end-frame_start+1, len(pb_names), 3))
center_arr = np.zeros((frame_end-frame_start+1, 1, 3))

for frame_idx in range(frame_start, frame_end+1):
    bpy.context.scene.frame_set(frame_idx)
    center_arr[frame_idx-frame_start, 0, :2] = pose_bones[center_name].head[:2]
    for bone_idx, pbn in enumerate(pb_names):
        joint_arr[frame_idx-frame_start, bone_idx, :] = pose_bones[pbn].head

data_dict = {'names': pb_names, 'data': joint_arr}

# Filter the chosen joints.
if output_flags['raw']:
    kp13_arr = pose_slice(parts_names_kp13, data_dict)
else:
    kp13_arr = joint_arr

# Normalize the selected joints' motion. 
kp13_arr_centered = kp13_arr - center_arr

########## Output files according to the flag setting ############
# Output the raw motion file. It contains all the possible joints 
# in your POSE model in every frame.
if output_flags['raw']:
    with open(op.join(output_dir, 'motion_dict.pkl'), 'wb') as f:
        pkl.dump(data_dict, f)

# Output the 13 keypoints motion trajectories over all frames.
if output_flags['kp13']:
    data_dict_kp13 = {'names': parts_names_kp13, 'data': kp13_arr}
    with open(op.join(output_dir, 'motion_dict_kp13.pkl'), 'wb') as f:
        pkl.dump(data_dict_kp13, f)

# Output the normalized 13 keypoints motion trajectories over all frames.
if output_flags['kp13_normed']:
    normed_data = kp13_arr_centered
    if norm_level >= 1:
        lowest_foot = kp13_arr_centered[:,-2:,-1:].min(axis=1)
        normed_data[:,:,-1] -= lowest_foot
    if norm_level == 2:
        highest_head = normed_data[:,0,-1].max()
        normed_data /= highest_head

    data_dict_kp13_normed = {'names': parts_names_kp13, 'data': normed_data}
    with open(op.join(output_dir, f'motion_dict_kp13_normed_lv{norm_level}.pkl'), 'wb') as f:
        pkl.dump(data_dict_kp13_normed, f)