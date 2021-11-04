import io
import os
import cv2
import argparse
import numpy as np
import pickle as pkl
import os.path as op
import matplotlib
from matplotlib import pyplot as plt
matplotlib.use("Agg")

from vis_utils import plot_skeleton_3d

# Save the current matplotlib buffer image to a numpy array.
def plt2arr(fig, dpi=180):
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

# Calculate the frame data of frame i
def skeleton_frame(i, view_angle=240, cam_height=10, limits=None):
    fig = plot_skeleton_3d(kp13_arr[i], view_angle, cam_height, ret_fig=True, limits=limits)
    data = plt2arr(fig, True)
    # plt.clf()
    plt.close('all')
    # plt.cla()
    return data

# Generate a video from the input absolute keypoints coordinate motion file
def generate_video(out_video_path, frame_num, fps=30, frame_step=1, view_angle=240, cam_height=10, limits=None):
    frame = skeleton_frame(0, limits=limits)
    out_size = frame.shape[:2]

    out = cv2.VideoWriter(out_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (out_size[1], out_size[0]))
    for i in range(0, frame_num, frame_step):
        frame = skeleton_frame(i, view_angle=view_angle, cam_height=cam_height, limits=limits)
        # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame)
    out.release()

# Generate an image sequence from the input absolute keypoints coordinate motion file
def generate_image_sequence(out_root, frame_num, frame_step=1, view_angle=240, cam_height=10, limits=None):
    for i in range(0, frame_num, frame_step):
        plot_skeleton_3d(kp13_arr[i], view_angle, cam_height, ret_fig=False, limits=limits)
        plt.savefig(op.join(out_root, '%5d.png'%i))
        plt.close('all')

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--in_path', type=str, help='Input motion dictionary pickle file path.')
    parser.add_argument('-f', '--fps', type=int, default=30, help='FPS of the output video.')
    parser.add_argument('-v', '--view_angle', type=int, default=240, help='View angle of the camera.')
    parser.add_argument('-c', '--cam_height', type=int, default=10, help='Height of the camera.')
    parser.add_argument('-s', '--frame_step', type=int, default=1, help='The number of frame stride when making the video.')
    parser.add_argument('-n', '--frame_num', type=int, help='For checking the contents. Only output the first n frames.')
    parser.add_argument('--seq', action='store_true', help='Output images sequence instead of video.')
    parser.add_argument('--norm', action='store_true', help='Input pose file is normalized.')
     
    args = parser.parse_args()

    root = op.split(args.in_path)[0]
    stem = op.splitext(op.split(args.in_path)[1])[0]
    out_root = op.join(root, 'frames')
    out_video_path = op.join(root, stem+'.mp4')
    if args.norm:
        limits = [[-0.5, 0.5], [-0.5, 0.5], [0, 1.2]]
    else:
        limits = [[-3, 3], [-3, 3], [0, 15]]
    
    with open(args.in_path, 'rb') as f:
        data_dict = pkl.load(f)

    kp13_arr = data_dict['data']
    frame_num = kp13_arr.shape[0] if args.frame_num is None else args.frame_num
    print("The shape of pose keypoints: ", kp13_arr.shape)

    if args.seq:
        if not op.exists(out_root):
            os.makedirs(out_root, exist_ok=True)
        generate_image_sequence(out_root, frame_num, args.frame_step, args.view_angle, args.cam_height, limits=limits)
    else:
        generate_video(out_video_path, frame_num, args.fps, args.frame_step, args.view_angle, args.cam_height, limits=limits)

    

    
