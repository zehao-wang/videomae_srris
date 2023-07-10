import argparse
import torch
from tqdm import tqdm
import pathlib
from utils.extractor import Extractor
import numpy as np
import os

parser = argparse.ArgumentParser(description='arguments')
parser.add_argument('--sample_rate', type=int, default=10, help='')
parser.add_argument('--pretrained_task', type=str, default="ssv2", choices=['ssv2', 'kinetics'],help='') 
parser.add_argument('--dataset_root_path', type=str, default="../data/General/processed_split/", help='')
parser.add_argument('--out_root', type=str, default='../data/General/processed_split/feats', help='')
args = parser.parse_args()

cam_types = ['cam1', 'cam2']


if __name__=="__main__":
    label2id={'Checking_Temperature': 0, 'Cleaning_Plate': 1, 'Closing_Clamps': 2, 'Closing_Doors': 3, 'Opening_Clamps': 4, 'Opening_Doors': 5, 'Putting_Plate_Back': 6, 'Removing_Plate': 7}
    
    extractor = Extractor(pretrain_name=args.pretrained_task, label2id=label2id, sample_rate=args.sample_rate)
    dataset_root_path = pathlib.Path(args.dataset_root_path)

    for split in ['val', 'train', 'test']:

        for cam_type in cam_types:
            all_states = []
            all_states_raw = []
            all_labels = []
            for vid_path in tqdm(list(dataset_root_path.glob(f"{cam_type}/{split}/*/*.mp4"))):
                feat_list, info_dict = extractor.extract(str(vid_path))
                if len(feat_list) > 1:
                    feat = np.mean(feat_list, axis=0)
                else:
                    feat = feat_list[0]
                label_id = info_dict['label']
                
                all_states.append(feat)
                all_labels.append(label_id)
                all_states_raw.append(feat_list)
            all_states_copy = np.stack(all_states)
            all_labels_copy = np.stack(all_labels)
            os.makedirs(f'./{args.out_root}/{cam_type}', exist_ok=True)
            np.save(f'./{args.out_root}/{cam_type}/{split}.npy', {"hidden": all_states_copy, "hidden_raw": all_states_raw, "labels": all_labels_copy, "id2label": extractor.id2label})
            print() 
