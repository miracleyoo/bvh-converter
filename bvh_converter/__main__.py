from __future__ import print_function, division
import sys
import csv
import argparse
import os
import io
import pandas as pd
import json

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
    
def filter_columns(file_in, file_out):
    parts_name = ['頭', '腕.L', '腕.R', 'ひじ.L', 'ひじ.R', '手首.L', '手首.R', '足.L', '足.R', 'ひざ.L', 'ひざ.R', '足首.L', '足首.R']
    col_names = []
    _ = [col_names.extend((k+'.X', k+'.Y', k+'.Z')) for k in parts_name]
    raw = pd.read_csv(file_in, encoding='gbk')
    selected_joints = raw[col_names]
    joints_dict = {}
    for part in parts_name:
        joint = selected_joints[[part+'.X', part+'.Y', part+'.Z']]
        joints_dict[part] = joint.to_numpy()
    if file_out is None:
        file_out = file_in[:-4]+'_kp13.json'
    with open(file_out, 'w') as f:
        json.dump(joints_dict, f)
    

def main():
    parser = argparse.ArgumentParser(
        description="Extract joint location and optionally rotation data from BVH file format.")
    parser.add_argument("filename", type=str, help='BVH file for conversion.')
    parser.add_argument("-r", "--rotation", action='store_true', help='Write rotations to CSV as well.')
    parser.add_argument("-m", "--mmd", action='store_true', help='Come from a MMD exported file.')
    parser.add_argument("-s", "--select", action='store_true', help='Select the specified columns.')
    
    args = parser.parse_args()

    file_in = args.filename
    do_rotations = args.rotation

    if not os.path.exists(file_in):
        print("Error: file {} not found.".format(file_in))
        sys.exit(0)
    print("Input filename: {}".format(file_in))

    other_s = process_bvhfile(file_in, args.mmd)

    print("Analyzing frames...")
    for i in range(other_s.frames):
        new_frame = process_bvhkeyframe(other_s.keyframes[i], other_s.root,
                                        other_s.dt * i)
    print("done")
    
    file_out = file_in[:-4] + "_worldpos.csv"

    with open_csv(file_out, 'w') as f:
        writer = csv.writer(f)
        header, frames = other_s.get_frames_worldpos()
        writer.writerow(header)
        for frame in frames:
            writer.writerow(frame)
    print("World Positions Output file: {}".format(file_out))

    if do_rotations:
        file_out = file_in[:-4] + "_rotations.csv"
    
        with open_csv(file_out, 'w') as f:
            writer = csv.writer(f)
            header, frames = other_s.get_frames_rotations()
            writer.writerow(header)
            for frame in frames:
                writer.writerow(frame)
        print("Rotations Output file: {}".format(file_out))


if __name__ == "__main__":
    main()
