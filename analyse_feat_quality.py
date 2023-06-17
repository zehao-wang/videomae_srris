
# model_ckpt = "MCG-NJU/videomae-base" # pre-trained model from which to fine-tune
model_ckpt = "MCG-NJU/videomae-base-finetuned-ssv2" # pre-trained model from which to fine-tune
batch_size = 4 # batch size for training and evaluation
# NOTE: use cam
dataset_root_path = "../data/General/processed_split/"
cam_type="cam2"

import torch
from collections import defaultdict
import pytorchvideo.data
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    Normalize,
    RandomShortSideScale,
    RemoveKey,
    ShortSideScale,
    UniformTemporalSubsample,
)

from torchvision.transforms import (
    Compose,
    Lambda,
    RandomCrop,
    CenterCrop,
    RandomHorizontalFlip,
    Resize,
)

import os
import pathlib
from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification

import imageio
import numpy as np
from transformers import TrainingArguments, Trainer
import evaluate
from torch.utils.data import DataLoader

dataset_root_path = pathlib.Path(dataset_root_path)
video_count_train = len(list(dataset_root_path.glob(f"{cam_type}/train/*/*.mp4")))
video_count_val = len(list(dataset_root_path.glob(f"{cam_type}/val/*/*.mp4")))
video_count_test = len(list(dataset_root_path.glob(f"{cam_type}/test/*/*.mp4")))
video_total = video_count_train + video_count_val + video_count_test
print(f"Total videos: {video_total}")

all_video_file_paths = (
    list(dataset_root_path.glob(f"{cam_type}/train/*/*.mp4"))
    + list(dataset_root_path.glob(f"{cam_type}/val/*/*.mp4"))
    + list(dataset_root_path.glob(f"{cam_type}/test/*/*.mp4"))
)

print('\033[1;32m [INFO]\033[0m sampled video paths', all_video_file_paths[:5])

class_labels = sorted({str(path).split("/")[-2] for path in all_video_file_paths})
label2id = {label: i for i, label in enumerate(class_labels)}
id2label = {i: label for label, i in label2id.items()}

print(f"Unique classes: {list(label2id.keys())}.")

image_processor = VideoMAEImageProcessor.from_pretrained(model_ckpt)
model = VideoMAEForVideoClassification.from_pretrained(
    model_ckpt,
    label2id=label2id,
    id2label=id2label,
    ignore_mismatched_sizes=True,  # provide this in case you're planning to fine-tune an already fine-tuned checkpoint
)

mean = image_processor.image_mean
std = image_processor.image_std
if "shortest_edge" in image_processor.size:
    height = width = image_processor.size["shortest_edge"]
else:
    height = image_processor.size["height"]
    width = image_processor.size["width"]
resize_to = (height, width)

num_frames_to_sample = model.config.num_frames
sample_rate = 20
fps = 30
clip_duration = num_frames_to_sample * sample_rate / fps


print("Number of frames to sample: ", num_frames_to_sample)
# Training dataset transformations. 
train_transform = Compose(
    [
        ApplyTransformToKey(
            key="video",
            transform=Compose(
                [
                    UniformTemporalSubsample(num_frames_to_sample),
                    Lambda(lambda x: x / 255.0),
                    Normalize(mean, std),
                    # RandomShortSideScale(min_size=256, max_size=320),
                    # RandomCrop(resize_to),
                    RandomShortSideScale(min_size=225, max_size=280),
                    CenterCrop(resize_to),
                    RandomHorizontalFlip(p=0.5),
                ]
            ),
        ),
    ]
)


# Training dataset.
# train_dataset = pytorchvideo.data.Ucf101(
#     data_path=os.path.join(dataset_root_path, f"{cam_type}/train"),
#     clip_sampler=pytorchvideo.data.make_clip_sampler("random", clip_duration),
#     decode_audio=False,
#     transform=train_transform,
# )

# Validation and evaluation datasets' transformations.
val_transform = Compose(
    [
        ApplyTransformToKey(
            key="video",
            transform=Compose(
                [
                    UniformTemporalSubsample(num_frames_to_sample),
                    Lambda(lambda x: x / 255.0),
                    Normalize(mean, std),
                    Resize(resize_to),
                ]
            ),
        ),
    ]
)

train_dataset = pytorchvideo.data.Ucf101(
    data_path=os.path.join(dataset_root_path, f"{cam_type}/train"),
    clip_sampler=pytorchvideo.data.make_clip_sampler("uniform", clip_duration),
    decode_audio=False,
    transform=val_transform,
)

# Validation and evaluation datasets.
val_dataset = pytorchvideo.data.Ucf101(
    data_path=os.path.join(dataset_root_path, f"{cam_type}/val"),
    clip_sampler=pytorchvideo.data.make_clip_sampler("uniform", clip_duration),
    decode_audio=False,
    transform=val_transform,
)

test_dataset = pytorchvideo.data.Ucf101(
    data_path=os.path.join(dataset_root_path, f"{cam_type}/test"),
    clip_sampler=pytorchvideo.data.make_clip_sampler("uniform", clip_duration),
    decode_audio=False,
    transform=val_transform,
)

# We can access the `num_videos` argument to know the number of videos we have in the
# dataset.
print("\n\033[1;32m [INFO]\033[0m dataset statistics (train, val, test):\n", train_dataset.num_videos, val_dataset.num_videos, test_dataset.num_videos, "\n")

sample_video = next(iter(train_dataset))

print("\033[1;32m [INFO]\033[0m dataset success, a batch dict contains: ", sample_video.keys())

def investigate_video(sample_video):
    """Utility to investigate the keys present in a single video sample."""
    for k in sample_video:
        if k == "video":
            print(k, sample_video["video"].shape)
        else:
            print(k, sample_video[k])

    print(f"Video label: {id2label[sample_video[k]]}")


investigate_video(sample_video)

def unnormalize_img(img):
    """Un-normalizes the image pixels."""
    img = (img * std) + mean
    img = (img * 255).astype("uint8")
    return img.clip(0, 255)


def create_gif(video_tensor, filename="sample.gif"):
    """Prepares a GIF from a video tensor.
    
    The video tensor is expected to have the following shape:
    (num_frames, num_channels, height, width).
    """
    video_tensor = video_tensor.permute(1, 0, 2, 3)
    frames = []
    for video_frame in video_tensor:
        frame_unnormalized = unnormalize_img(video_frame.permute(1, 2, 0).numpy())
        frames.append(frame_unnormalized)
    kargs = {"duration": 0.25}
    imageio.mimsave(filename, frames, "GIF", **kargs)

video_tensor = sample_video["video"]
create_gif(video_tensor)
print("\033[1;32m [INFO]\033[0m Please check sample.gif for processed video clip samples")

def collate_fn(examples):
    """The collation function to be used by `Trainer` to prepare data batches."""
    # permute to (num_frames, num_channels, height, width)
    pixel_values = torch.stack(
        [example["video"].permute(1, 0, 2, 3) for example in examples]
    )
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}

dataloader = DataLoader(train_dataset,batch_size=32,collate_fn=collate_fn,num_workers=0,pin_memory=False,)

print("\033[1;32m [INFO]\033[0m We are now trying to plot the feature correlation")


all_states = []
all_labels = []
cnt = 0
model.eval()
with torch.no_grad():
    for step, inputs in enumerate(dataloader):
        labels = inputs.pop("labels")
        outputs = model.videomae(**inputs)
        # import ipdb;ipdb.set_trace() # breakpoint 222
        # all_states.append(outputs['last_hidden_state'][0][:, 0].detach().cpu().numpy())
        all_states.append(outputs['last_hidden_state'].detach().cpu().numpy())
        all_labels.append(labels.detach().cpu().numpy())
        cnt += len(labels)
        print(cnt)


import matplotlib.pyplot as plt
all_states_copy = np.concatenate([stat[:, 0, :] for stat in all_states], axis=0)
all_labels_copy = np.concatenate(all_labels, axis=0)

labels_unique = np.unique(all_labels_copy)

from sklearn.metrics.pairwise import cosine_similarity

def plt_corr():
    X = []
    for la in labels_unique:
        ins = np.where(all_labels_copy == la)
        feats = all_states_copy[ins]
        # X.append(feats[0])
        # print(feats.shape)
        X.append(np.mean(feats, axis=0))

    mat = cosine_similarity(X,X)
    x_ticks = ['Checking_Temperature', 'Cleaning_Plate', 'Closing_Clamps', 'Closing_Doors', 'Opening_Clamps', 'Opening_Doors', 'Putting_Plate_Back', 'Removing_Plate']

    plt.matshow(mat)
    plt.xticks(np.arange(len(X)), x_ticks, rotation=90)
    plt.yticks(np.arange(len(X)), x_ticks, rotation=0)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig('corr.png', dpi=300, bbox_inches='tight')