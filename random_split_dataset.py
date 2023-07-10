import pathlib
import random
import os
import shutil
import argparse

parser = argparse.ArgumentParser(description='arguments')
parser.add_argument('--dataset_root_path', type=str, default="../data/General/processed/", help='')
parser.add_argument('--dataset_out_root_path', type=str, default="../data/General/processed_split/", help='')
args = parser.parse_args()

test_set_id = [12,14,15,16]
cam_type="cam1"
percent = [0.7,0.3] # train, val

if __name__ == '__main__':
    for cam_type in ['cam1', 'cam2']:
        dataset_root_path = os.path.join(args.dataset_root_path, cam_type)
        tran_files = []
        val_files  = []
        test_files = []
        for act_type in os.listdir(dataset_root_path):
            file_paths = os.listdir(os.path.join(dataset_root_path, act_type))
            random.shuffle(file_paths)
            file_paths = [os.path.join(dataset_root_path, act_type, fname) for fname in file_paths]

            # filter out test files
            tests = []
            others = []
            for file_path in file_paths:
                file_id = int(file_path.split('/')[-1].split("_")[0])
                if file_id in test_set_id:
                    tests.append(file_path)
                else:
                    others.append(file_path)

            anchors= [int(len(others) * percent[0])]
            tran_files += others[:anchors[0]]
            val_files += others[anchors[0]: ]
            test_files += tests

        print(f'\033[1;32m [INFO]\033[0m Splits count: train: {len(tran_files)}, val: {len(val_files)}, test: {len(test_files)}')
        files = {'train': tran_files, 'val':val_files, 'test':test_files}
        for split, fs in files.items():
            for file in fs:
                src_fpath = file
                dest_fpath = str(file).replace('processed', 'processed_split').replace(cam_type, cam_type+'/'+split)
                os.makedirs(os.path.dirname(dest_fpath), exist_ok=True)
                shutil.copy(src_fpath, dest_fpath)
