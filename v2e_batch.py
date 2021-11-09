import os
import glob
import subprocess
import os.path as op
from pathlib2 import Path
from string import Template

V2E_PATH = r'E:\GitHub\v2e'
DATA_ROOT = r'J:\datasets\DVS\mmd'
TYPE_NAME = 'dvs346_clean_raw'

os.chdir(V2E_PATH)

# CMD = r"""python v2e.py --overwrite --unique_output_folder true --no_preview 
# --input_slowmotion_factor 1 --auto_timestamp_resolution true 
# --pos_thres 0.2 --neg_thres 0.2 --sigma_thres 0.03 --cutoff_hz 30 
# --leak_rate_hz 0.01 --shot_noise_rate_hz 0.5 --output_height 260 
# --output_width 462 --dvs_exposure duration 0.0166666667 
# --timestamp_resolution 0.002 --dvs_h5 events.h5 --dvs_text None 
# --dvs_aedat2 None --avi_frame_rate 60 --slomo_model "input/SuperSloMo39.ckpt" 
# --input "$input" --output_folder "$output" --batch_size 16
# """.replace('\n',' ').replace('  ', ' ')

### Noisy 16:9 original aspect ratio output
# CMD = r"""python v2e.py --overwrite --unique_output_folder true --no_preview 
# --input_slowmotion_factor 1 --auto_timestamp_resolution false 
# --pos_thres 0.2 --neg_thres 0.2 --sigma_thres 0.03 --cutoff_hz 30 
# --leak_rate_hz 0.01 --shot_noise_rate_hz 0.5 --output_height 260 
# --output_width 462 --dvs_exposure duration 0.0166666667 --disable_slomo 
# --dvs_h5 events.h5 --dvs_text None --dvs_aedat2 None --avi_frame_rate 60 
# --input "$input" --output_folder "$output" --batch_size 16
# """.replace('\n',' ').replace('  ', ' ')

## Clean dvs346 style output
CMD = r"""python v2e.py --overwrite --unique_output_folder true --no_preview 
--input_slowmotion_factor 1 --auto_timestamp_resolution false 
--pos_thres 0.2 --neg_thres 0.2 --sigma_thres 0.03 --dvs346 --dvs_params clean
--dvs_exposure duration 0.0166666667 --disable_slomo 
--dvs_h5 events.h5 --dvs_text events.txt --dvs_aedat2 events.aedat2 
--avi_frame_rate 60 
--input "$input" --output_folder "$output" --batch_size 16
""".replace('\n',' ').replace('  ', ' ')

CMD_TEMPLATE = Template(CMD)

# print(CMD_TEMPLATE.substitute(input=r"D:\MMD\Projects\low-ligth-multi-motion-test\001.avi", output=r"D:\MMD\Projects\low-ligth-multi-motion-test\test17_mid_noise"))

# Calculate the V2E Results for each valid compressed video file
def cal_v2e(root, v2e_root, test_only=True):
    raw_videos = glob.glob(op.join(root, '*', '*.avi'))
    os.makedirs(v2e_root, exist_ok=True)
    print('Total Video Num: ', len(raw_videos))

    for i, raw_path in enumerate(raw_videos):
        raw_dirname = Path(raw_path).parts[-2]
        stem = Path(raw_path).stem
        new_dir = op.join(v2e_root, raw_dirname, stem)
        # new_path = op.join(new_dir, filename)
        if not op.exists(new_dir):
            os.makedirs(new_dir, exist_ok=True)
        print(f'[{i+1}/{len(raw_videos)}]: {raw_path} -> {new_dir}')
        if not test_only:
            command = CMD_TEMPLATE.substitute(input=raw_path, output=new_dir)
            # print(command)
            os.system(command)
            # os.rename(raw_path, new_dir)


comp_root = op.join(DATA_ROOT, 'raw')
v2e_root = op.join(DATA_ROOT, 'v2e', TYPE_NAME)
cal_v2e(comp_root, v2e_root, test_only=False)

# os.system(CMD)
# input_files = glob.glob(op.join(DATA_ROOT, 'compressed', '*', '*.mp4'))
# print(input_files)
# os.system('dir')
# os.system('python v2e.py')
