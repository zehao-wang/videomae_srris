
# model_ckpt = "MCG-NJU/videomae-base" # pre-trained model from which to fine-tune
model_ckpt = "MCG-NJU/videomae-base-finetuned-ssv2" # pre-trained model from which to fine-tune
batch_size = 4 # batch size for training and evaluation
import torch
from collections import defaultdict
# dataset_root_path = "../data/UCF101_subset/"
# import pathlib
# dataset_root_path = pathlib.Path(dataset_root_path)
# video_count_train = len(list(dataset_root_path.glob("train/*/*.avi")))
# video_count_val = len(list(dataset_root_path.glob("val/*/*.avi")))
# video_count_test = len(list(dataset_root_path.glob("test/*/*.avi")))
# video_total = video_count_train + video_count_val + video_count_test
# print(f"Total videos: {video_total}")

# all_video_file_paths = (
#     list(dataset_root_path.glob("train/*/*.avi"))
#     + list(dataset_root_path.glob("val/*/*.avi"))
#     + list(dataset_root_path.glob("test/*/*.avi"))
# )

# NOTE: use cam
dataset_root_path = "../data/General/processed_split/"
cam_type="cam1"
import pathlib
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


from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification
image_processor = VideoMAEImageProcessor.from_pretrained(model_ckpt)
model = VideoMAEForVideoClassification.from_pretrained(
    model_ckpt,
    label2id=label2id,
    id2label=id2label,
    ignore_mismatched_sizes=True,  # provide this in case you're planning to fine-tune an already fine-tuned checkpoint
)

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
mean = image_processor.image_mean
std = image_processor.image_std
if "shortest_edge" in image_processor.size:
    height = width = image_processor.size["shortest_edge"]
else:
    height = image_processor.size["height"]
    width = image_processor.size["width"]
resize_to = (height, width)

num_frames_to_sample = model.config.num_frames
sample_rate = 4
fps = 30
clip_duration = num_frames_to_sample * sample_rate / fps


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
train_dataset = pytorchvideo.data.Ucf101(
    data_path=os.path.join(dataset_root_path, f"{cam_type}/train"),
    clip_sampler=pytorchvideo.data.make_clip_sampler("random", clip_duration),
    decode_audio=False,
    transform=train_transform,
)

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

import imageio
import numpy as np
from IPython.display import Image


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


from transformers import TrainingArguments, Trainer
model_name = model_ckpt.split("/")[-1]
new_model_name = f"outputs/{model_name}-finetuned-ucf101-subset"
num_epochs = 4

args = TrainingArguments(
    new_model_name,
    remove_unused_columns=False,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    warmup_ratio=0.1,
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    push_to_hub=False,
    max_steps=(train_dataset.num_videos // batch_size) * num_epochs,
)

import evaluate
metric = evaluate.load("accuracy")

# the compute_metrics function takes a Named Tuple as input:
# predictions, which are the logits of the model as Numpy arrays,
# and label_ids, which are the ground-truth labels as Numpy arrays.
def compute_metrics(eval_pred):
    """Computes accuracy on a batch of predictions."""
    predictions = np.argmax(eval_pred.predictions, axis=1)
    metric_dict = metric.compute(predictions=predictions, references=eval_pred.label_ids)

    tot = defaultdict(int)
    correct = defaultdict(int) 
    for pred_id, gt_id in zip(predictions, eval_pred.label_ids):
        tot[gt_id] += 1
        if pred_id == gt_id:
            correct[gt_id] += 1
    per_cls_acc = []
    for k in tot.keys():
        per_cls_acc.append((id2label[k], correct[k]/tot[k], correct[k], tot[k]))
    metric_dict.update({"per_cls_acc": per_cls_acc})
    
    print(metric_dict)
    return metric_dict

import torch

def collate_fn(examples):
    """The collation function to be used by `Trainer` to prepare data batches."""
    # permute to (num_frames, num_channels, height, width)
    pixel_values = torch.stack(
        [example["video"].permute(1, 0, 2, 3) for example in examples]
    )
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}


trainer = Trainer(
    model,
    args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=image_processor,
    compute_metrics=compute_metrics,
    data_collator=collate_fn,
)

train_results = trainer.train()
print("\033[1;32m [INFO]\033[0m Train successfully!")

trainer.evaluate(test_dataset)

trainer.save_model()
test_results = trainer.evaluate(test_dataset)
trainer.log_metrics("test", test_results)
trainer.save_metrics("test", test_results)
trainer.save_state()
print("\033[1;32m [INFO]\033[0m Evaluation and checkpt save successfully!")