import pathlib
import random
import os
import shutil

dataset_root_path = "../data/General/processed/"
dataset_out_root_path = "../data/General/processed_split/"
cam_type="cam1"
percent = [0.6,0.2,0.2]

dataset_root_path = os.path.join(dataset_root_path, cam_type)
tran_files = []
val_files  = []
test_files = []
for act_type in os.listdir(dataset_root_path):
    file_paths = os.listdir(os.path.join(dataset_root_path, act_type))
    random.shuffle(file_paths)
    file_paths = [os.path.join(dataset_root_path, act_type, fname) for fname in file_paths]
    anchors= [int(len(file_paths) * percent[0]), int(len(file_paths) * (percent[0]+percent[1])) ]
    tran_files += file_paths[:anchors[0]]
    val_files += file_paths[anchors[0]: anchors[1]]
    test_files += file_paths[anchors[1]: ]

print(f'\033[1;32m [INFO]\033[0m Splits count: train: {len(tran_files)}, val: {len(val_files)}, test: {len(test_files)}')
files = {'train': tran_files, 'val':val_files, 'test':test_files}
for split, fs in files.items():
    for file in fs:
        src_fpath = file
        dest_fpath = str(file).replace('processed', 'processed_split').replace(cam_type, cam_type+'/'+split)
        os.makedirs(os.path.dirname(dest_fpath), exist_ok=True)
        shutil.copy(src_fpath, dest_fpath)


# dataset_root_path = pathlib.Path(dataset_root_path)

# all_video_file_paths = (
#     list(dataset_root_path.glob(f"{cam_type}/*/*.mp4"))
# )
# print(len(all_video_file_paths))
# random.shuffle(all_video_file_paths)

# # TODO: might have zero-shot case in val and test, try to do label-wise split in the future
# anchors= [int(len(all_video_file_paths) * percent[0]), int(len(all_video_file_paths) * (percent[0]+percent[1])) ]
# tran_files = all_video_file_paths[:anchors[0]]
# val_files = all_video_file_paths[anchors[0]: anchors[1]]
# test_files = all_video_file_paths[anchors[1]: ]

# print(f"Splits count: train: {len(tran_files)}, val: {len(val_files)}, test: {len(test_files)}")

# files = {'train': tran_files, 'val':val_files, 'test':test_files}
# for split, fs in files.items():
#     for file in fs:
#         src_fpath = file
#         dest_fpath = str(file).replace('processed', 'processed_split').replace(cam_type, cam_type+'/'+split)
#         os.makedirs(os.path.dirname(dest_fpath), exist_ok=True)
#         shutil.copy(src_fpath, dest_fpath)