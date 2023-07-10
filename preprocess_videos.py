from mmaction.utils import frame_extract
import tempfile
import json
import os
import numpy as np
import moviepy.editor as mpy
import cv2
import mmcv
from tqdm import tqdm
import random
import copy
from multiprocessing import Pool
import argparse


parser = argparse.ArgumentParser(description='arguments')
parser.add_argument('--frame_shift', type=int, default=5, help='small frame padding extension to handle annotation error')
parser.add_argument('--data_root', type=str, default='../data/General', help='')
parser.add_argument('--out_root', type=str, default='../data/General/processed', help='')
args = parser.parse_args()

short_side = None
CAM_MAPPING = {"Camera1_anonymized.mp4": "cam1", "Camera2_anonymized.mp4":"cam2", 'SmartGlasses_anonymized.mp4':'ego'}
def center_crop(img, dim):
	"""Returns center cropped image
	Args:
	img: image to be center cropped
	dim: dimensions (width, height) to be cropped
	"""
	width, height = img.shape[1], img.shape[0]

	# process crop width and height for max available dimension
	crop_width = dim[0] if dim[0]<img.shape[1] else img.shape[1]
	crop_height = dim[1] if dim[1]<img.shape[0] else img.shape[0] 
	mid_x, mid_y = int(width/2), int(height/2)
	cw2, ch2 = int(crop_width/2), int(crop_height/2) 
	crop_img = img[mid_y-ch2:mid_y+ch2, mid_x-cw2:mid_x+cw2]
	return crop_img

def process_video(video_path, duration=None):
    # video_path = './data/sirris/Videos/002 Normal scenario Operator1/Camera1_anonymized.mp4'
    tmp_dir = tempfile.TemporaryDirectory()
    frame_paths, frames = \
        frame_extract(video_path, short_side, tmp_dir.name)
    
    # for i, frame in enumerate(frames):
    #     frame = cv2.resize(frame, (0,0), fx=0.25, fy=0.25) 
    #     short = min(frame.shape[0], frame.shape[1])
    #     frames[i] = center_crop(frame, (short, short))

    timestamps=[0]
    if duration is not None:
        time_per_frame = duration/(len(frames)-1)
        for i in range(1, len(frames)):
            timestamps.append(timestamps[-1] + time_per_frame)
        timestamps = np.array([np.round(timestamp).astype(int) for timestamp in timestamps])

    return frames, timestamps

def visualize_and_dump(frames_dict, annts, dump_dir, prefix="", shift=5):
    """
    Args:
        frames:
        annts:
        dump_dir:
    """
    print("START visualization")
    for j, (class_id, annt) in enumerate(annts.items()):
        if not annt['is_event']:
            continue

        class_name = annt['class_name']
        frame_indies = annt['frame_idx']
        if len(frame_indies) == 0:
            continue

        start_idx = min(frame_indies)
        end_idx = max(frame_indies)
        
        bbox_s = np.round(annt['bbox'][0][:2]).astype(int)
        bbox_e = np.round(annt['bbox'][0][2:]).astype(int)
        
        vid_idx = prefix + f"_{j}"
        for key in frames_dict.keys():
            cam_type = CAM_MAPPING[key]
            out_dir = os.path.join(dump_dir, cam_type, class_name.replace(' ', '_'))
            os.makedirs(out_dir, exist_ok=True)

            frames_full = frames_dict[key]['frames']
            # frames_full = copy.deepcopy(frames_dict[key]['frames'])
            # for i, frame in enumerate(frames[max(start_idx-shift, 0): min(end_idx+1+shift, len(frames)-1)]):
            #     cv2.imwrite(os.path.join(cam_out_dir, f"{i}.png"), frame[:, :, ::-1])
            # FONTSCALE = 1.5
            # FONTFACE = cv2.FONT_HERSHEY_DUPLEX
            # THICKNESS = 2
            # LINETYPE = 2

            # for frame in frames[start_idx: end_idx+1]: 
            #     frame = cv2.putText(frame, class_name, (bbox_s[0], bbox_s[1]+ 80), FONTFACE, FONTSCALE,
            #         (0, 0, 255), THICKNESS, LINETYPE)
            #     frame = cv2.rectangle(frame, bbox_s, bbox_e, (0,0,255), THICKNESS)

            frames = [frame[:, :, ::-1] for frame in frames_full[max(start_idx-shift, 0): min(end_idx+1+shift, len(frames_full)-1)]] 
            vid = mpy.ImageSequenceClip(frames, fps=25)
            # vid.write_videofile(os.path.join(out_dir, f"{vid_idx}.avi"), codec='png',remove_temp=True)
            vid.write_videofile(os.path.join(out_dir, f"{vid_idx}.mp4"), remove_temp=True)

def run(args, folder_name):
    idx = folder_name.split()[0]

    # video_files = ['Camera1_anonymized.mp4', 'Camera2_anonymized.mp4', 'SmartGlasses_anonymized.mp4']
    video_files = ['Camera1_anonymized.mp4', 'Camera2_anonymized.mp4']

    meta_files = []
    for file in os.listdir(os.path.join(args.data_root, 'Annotations')):
        if file.startswith(idx):
            meta_files.append(file)

    for meta_file in sorted(meta_files):
        meta = json.load(open(os.path.join(args.data_root, 'Annotations', meta_file)))
        camera_id = meta['metadata']['name'].split('.')[0].split('_')[-1]
        duration = meta['metadata']['duration']

        main_cam = None
        print("Main: ", camera_id) 
        frame_dict = {}
        for video_file in video_files:
            if camera_id in video_file:
                frames, timestamps = process_video(os.path.join(args.data_root, 'Videos', folder_name, video_file), duration=duration)
                print(f"Main {video_file}: ", len(frames))
                frame_dict[video_file] = {"frames": frames, "timestamps": timestamps}
                main_cam = video_file
            else:
                frames, timestamps = process_video(os.path.join(args.data_root, 'Videos', folder_name, video_file))
                print(f"Others {video_file}: ", len(frames))
                frame_dict[video_file] = {"frames": frames}

        total_frames = len(frame_dict[main_cam]['frames'])
        timestamps = frame_dict[main_cam]['timestamps']
        for key in frame_dict.keys():
            if key != main_cam:
                frames = frame_dict[key]["frames"]
                samples = [frames[i] for i in sorted(np.random.choice(range(len(frames)), size=total_frames, replace=True))]
                frame_dict[key]["frames"] = samples
                print(len(samples))

        print()

        annotations = {} # frameidx: [x1,y1,x2,y2]
        for ins_id, instance in enumerate(tqdm(meta['instances'])):
            print(instance['meta']['type'], instance['meta']['className'], instance['meta']['start'], instance['meta']['end'])
            
            class_name = instance['meta']['className']
            class_ID = instance['meta']['classId']
            if len(instance['parameters']) > 1:
                import ipdb;ipdb.set_trace() # breakpoint 79

            # annotations[class_ID] = {"class_name": class_name, "frame_idx": [], "bbox": []}
            annotations[ins_id] = {"class_name": class_name, "frame_idx": [], "bbox": []}
            for annts in instance['parameters'][0]['timestamps']:
                if 'points' not in annts:
                    # import ipdb;ipdb.set_trace() # breakpoint 97
                    print(f"[WARNING] missing annotation for {instance['meta']['type']}, {instance['meta']['className']}")
                    
                timestamp = annts['timestamp']
                frame_idx = np.argmin(abs(timestamps-timestamp))
                # annotations[class_ID]['frame_idx'].append(frame_idx)
                # annotations[class_ID]['bbox'].append((annts['points']['x1'], annts['points']['y1'], annts['points']['x2'], annts['points']['y2']))
                
                annotations[ins_id]['frame_idx'].append(frame_idx)
                if instance['meta']['type'] == 'event':
                    annotations[ins_id]['bbox'].append((meta['metadata']['width']//2-200, 0, meta['metadata']['width']//2-199, 1))
                    annotations[ins_id]['is_event']=True
                else:
                    annotations[ins_id]['bbox'].append((annts['points']['x1'], annts['points']['y1'], annts['points']['x2'], annts['points']['y2']))
                    annotations[ins_id]['is_event']=False

        visualize_and_dump(frame_dict, annotations, os.path.join(args.out_root), prefix=idx, shift=args.frame_shift)

if __name__ == "__main__":
    # pool = Pool(4)
    for folder_name in os.listdir(os.path.join(args.data_root, 'Videos')):
        # pool.apply_async(run, args=(folder_name, ))
        run(args, folder_name)

        # import ipdb;ipdb.set_trace() # breakpoint 97
        # print()

    # pool.close()
    # pool.join()

