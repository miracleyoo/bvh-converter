from __future__ import print_function, division
import sys
import csv
import argparse
import os
import io
import numpy as np
import pandas as pd
import pickle as pkl
import os.path as op

from bvh_converter.bvhplayer_skeleton import process_bvhfile, process_bvhkeyframe

"""
Based on: http://www.dcs.shef.ac.uk/intranet/research/public/resmes/CS0111.pdf

Notes:
 - For each frame we have to recalculate from root
 - End sites are semi important (used to calculate length of the toe? vectors)
"""


def open_csv(filename, mode='r'):
    """Open a csv file in proper mode depending on Python version."""
    if sys.version_info < (3,):
        return io.open(filename, mode=mode+'b')
    else:
        return io.open(filename, mode=mode, newline='')
    
def extend_names_xyz(names):
    if type(names) != list:
        names = [str(names)]
    col_names = []
    _ = [col_names.extend((k+'.X', k+'.Y', k+'.Z')) for k in names]
    return col_names

def extract_keypoints(file_in, file_out=None, norm='ori', ret=False):
    assert norm in ['ori', 'norm', 'both']
    parts_names = ['頭', '腕.R', '腕.L', 'ひじ.R', 'ひじ.L', '足.R', '足.L', '手首.R', '手首.L',  'ひざ.R', 'ひざ.L', '足ＩＫ.R', '足ＩＫ.L']
    col_names = extend_names_xyz(parts_names)
    # Blender will add 5 extra frames to a vmd 
    # motion compared to original MMD motion.
    raw = pd.read_csv(file_in, encoding='gbk')[5:]
    selected_joints = raw[col_names]

    joint_list = []
    
    for part in parts_names:
        joint = selected_joints[[part+'.X', part+'.Y', part+'.Z']].to_numpy()
        joint_list.append(joint)

    joint_arr = np.array(joint_list)
    del joint_list

    if norm != 'ori':
        xy_norm_factor = joint_arr[:,:,:2].mean(axis=0)
        z_norm_factor = joint_arr[-2:,:,-1:].min(axis=0)
        norm_factor = np.concatenate((xy_norm_factor,z_norm_factor), axis=1)
        # print(norm_factor.shape)
        
        joint_arr_normed = joint_arr - norm_factor
        joint_arr_normed = joint_arr_normed.transpose(1,0,2)

    joint_arr = joint_arr.transpose(1,0,2)

    if file_out is None:
        if norm != 'norm':
            file_out = op.splitext(file_in)[0]+'_kp13.pkl'
            with open(file_out, 'wb') as f:
                pkl.dump(joint_arr, f)

        if norm != 'ori':
            file_out = op.splitext(file_in)[0]+'_kp13_normed.pkl'
            with open(file_out, 'wb') as f:
                pkl.dump(joint_arr_normed, f)

    if ret:
        if norm == 'ori':
            return joint_arr
        elif norm == 'norm':
            return joint_arr_normed
        else:
            return joint_arr, joint_arr_normed
    

def main():
    parser = argparse.ArgumentParser(
        description="Extract joint location and optionally rotation data from BVH file format.")
    parser.add_argument("filename", type=str, help='BVH file for conversion.')
    parser.add_argument("-r", "--rotation", action='store_true', help='Write rotations to CSV as well.')
    parser.add_argument("-d", "--debug", action='store_true', help='Debug Mode.')
    parser.add_argument("-m", "--mmd", action='store_true', help='Come from a MMD exported file.')
    parser.add_argument("-k", "--keypoints_extract", action='store_true', help='Whether to extract keypoints.')
    parser.add_argument("-n", "--norm_type", default='norm', choices=('ori', 'norm', 'both'), type=str, help='Normalization Type.')
    
    args = parser.parse_args()

    file_in = args.filename
    do_rotations = args.rotation

    if not os.path.exists(file_in):
        print("❀ Error: file {} not found.".format(file_in))
        sys.exit(0)
    print("❀ Input filename: {}".format(file_in))

    other_s = process_bvhfile(file_in, args.mmd, DEBUG=args.debug)

    print("❀ Analyzing frames...")
    for i in range(other_s.frames):
        new_frame = process_bvhkeyframe(other_s.keyframes[i], other_s.root,
                                        other_s.dt * i, DEBUG=args.debug)
    print("done")
    
    out_csv_path = file_in[:-4] + "_worldpos.csv"

    with open_csv(out_csv_path, 'w') as f:
        writer = csv.writer(f)
        header, frames = other_s.get_frames_worldpos()
        writer.writerow(header)
        for frame in frames:
            writer.writerow(frame)
    print("❀ World Positions Output file: {}".format(out_csv_path))

    if args.keypoints_extract:
        print("❀ Extracting Keypoints...")
        extract_keypoints(out_csv_path, norm=args.norm_type)
        print("done")

    if do_rotations:
        file_out = file_in[:-4] + "_rotations.csv"
    
        with open_csv(file_out, 'w') as f:
            writer = csv.writer(f)
            header, frames = other_s.get_frames_rotations()
            writer.writerow(header)
            for frame in frames:
                writer.writerow(frame)
        print("❀ Rotations Output file: {}".format(file_out))


if __name__ == "__main__":
    main()
