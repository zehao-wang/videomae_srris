from utils.extractor import Extractor
import numpy as np
import moviepy.editor as mpy
import argparse
import pathlib
import os
from utils.label_utils import get_meta, cloest_label
parser = argparse.ArgumentParser(description='arguments')
parser.add_argument('--sample_rate', type=int, default=10, help='')
parser.add_argument('--data_root', type=str, default='../data/General/Videos', help='')
parser.add_argument('--out_root', type=str, default='../data/General/processed_split/full', help='')
parser.add_argument('--for_vis', action='store_true', default=False)
args = parser.parse_args()
os.makedirs(os.path.join(args.out_root), exist_ok=True)

label2id={'Checking_Temperature': 0, 'Cleaning_Plate': 1, 'Closing_Clamps': 2, 'Closing_Doors': 3, 'Opening_Clamps': 4, 'Opening_Doors': 5, 'Putting_Plate_Back': 6, 'Removing_Plate': 7}

data_root = pathlib.Path(args.data_root)
paths = [path for path in list(data_root.glob(f"*/*.mp4")) if int(str(path).split('/')[-2].split(' ')[0]) in [12,14,15,16] ]

extractor = Extractor(pretrain_name='ssv2', label2id=label2id, sample_rate=args.sample_rate)

for file_path in paths:
    video_id = str(file_path).split('/')[-2].split(' ')[0]
    cam_type = str(file_path).split('/')[-1].split('_')[0]
    clip_list = extractor.get_clips(str(file_path), visualization=args.for_vis)
    for i, clip in enumerate(clip_list):
        frames = [np.transpose(frame, (1,2,0)) for frame in clip]
        vid = mpy.ImageSequenceClip(frames, fps=25)

        os.makedirs(os.path.join(args.out_root, video_id), exist_ok=True)
        os.makedirs(os.path.join(args.out_root, video_id, cam_type), exist_ok=True)
        vid.write_videofile(os.path.join(args.out_root, video_id, cam_type, f"{i}.mp4"), remove_temp=True)

print('\033[1;32m [INFO]\033[0m Start feature extraction')
for file_path in paths:
    video_id = str(file_path).split('/')[-2].split(' ')[0]
    cam_type = str(file_path).split('/')[-1].split('_')[0]
    event_seq = get_meta('../data/General/Annotations', video_id)

    feat_list, info_dict = extractor.extract(str(file_path), no_label=True)
    assert len(feat_list) == len(info_dict['clip_times'])

    labels = []
    for feat, clip_time in zip(feat_list, info_dict['clip_times']):
        label = cloest_label(clip_time, event_seq)
        labels.append(label)
    print("GT seq: ", labels)
    out_file = f'{args.out_root}/{video_id}/{cam_type}.npy'
    print(f'\033[1;32m [INFO]\033[0m Output to {out_file}')
    np.save(out_file, {"feat": feat_list, "labels": labels, "id2label": extractor.id2label, "label2id": label2id})
    