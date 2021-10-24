import bpy
import os.path as op
import numpy as np
import pickle as pkl
C = bpy.context
D = bpy.data

###################################################################
############### PARAMETERS YOU NEED TO FILL IN ####################
model_name='Tda Hood Miku 1.00 by iRon0129'
output_dir = r'D:\MMD\Setting_Pairs\miku_neutral_street_bluestar'
output_flags = {'raw': True, 'kp13': False, 'kp13_normed': True}
###################################################################

# Get part of the data using bone names
def pose_slice(names, data):
    if type(names) not in (list, tuple):
        names = [str(names)]
    indexes = np.array([data['names'].index(k) for k in names])
    joint_arr = data['data'][:,indexes]
    return joint_arr

armature_name = model_name + '_arm'
frame_start = C.scene.frame_start
frame_end = C.scene.frame_end

pose_bones = {pb.name: pb for pb in D.objects[armature_name].pose.bones}
pb_names = list(pose_bones.keys())

data = np.zeros((frame_end-frame_start+1, len(pb_names), 3))
for frame_idx in range(frame_start, frame_end+1):
    bpy.context.scene.frame_set(frame_idx)
    for bone_idx, pbn in enumerate(pb_names):
        data[frame_idx-1, bone_idx, :] = pose_bones[pbn].head

data_dict = {'names': pb_names, 'data': data}

parts_names_kp13 = ['頭', '腕.R', '腕.L', 'ひじ.R', 'ひじ.L', '足.R', '足.L', 
        '手首.R', '手首.L',  'ひざ.R', 'ひざ.L', '足首.R', '足首.L']
kp13_arr = pose_slice(parts_names_kp13, data_dict)
center_arr = pose_slice('グルーブ', data_dict)
center_arr[:,:,2] = 0
kp13_arr_normed = kp13_arr - center_arr

if output_flags['raw']:
    with open(op.join(output_dir, 'motion_dict.pkl'), 'wb') as f:
        pkl.dump(data_dict, f)

if output_flags['kp13']:
    data_dict_kp13 = {'names': parts_names_kp13, 'data': kp13_arr}
    with open(op.join(output_dir, 'motion_dict_kp13.pkl'), 'wb') as f:
        pkl.dump(data_dict_kp13, f)

if output_flags['kp13_normed']:
    data_dict_kp13 = {'names': parts_names_kp13, 'data': kp13_arr_normed}
    with open(op.join(output_dir, 'motion_dict_kp13_normed.pkl'), 'wb') as f:
        pkl.dump(data_dict, f)