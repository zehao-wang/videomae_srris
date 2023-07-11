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
from tqdm import tqdm
from pytorchvideo.data.video import VideoPathHandler

class Extractor(object):
    def __init__(self, pretrain_name='ssv2', label2id=None, sample_rate=10) -> None:
        if label2id is None:
            raise ValueError("Please provide label2id dict")
        id2label = {i: label for label, i in label2id.items()}
        self.label2id = label2id
        self.id2label = id2label

        #  model_ckpt = "MCG-NJU/videomae-base" # base model not finetune cls head
        if pretrain_name == 'ssv2':
            model_ckpt = "MCG-NJU/videomae-base-finetuned-ssv2"
        elif pretrain_name == 'kinetics':
            model_ckpt="MCG-NJU/videomae-base-finetuned-kinetics"
        else:
            raise NotImplementedError()
        
        # Init model
        self.image_processor = VideoMAEImageProcessor.from_pretrained(model_ckpt)
        self.model = VideoMAEForVideoClassification.from_pretrained(
            model_ckpt,
            label2id=label2id,
            id2label=id2label,
            ignore_mismatched_sizes=True,  # provide this in case you're planning to fine-tune an already fine-tuned checkpoint
        )

        # Init transforms
        mean = self.image_processor.image_mean
        std = self.image_processor.image_std
        if "shortest_edge" in self.image_processor.size:
            height = width = self.image_processor.size["shortest_edge"]
        else:
            height = self.image_processor.size["height"]
            width = self.image_processor.size["width"]
        resize_to = (height, width)
        self.resize_to = resize_to
        num_frames_to_sample = self.model.config.num_frames

        # self.transform = Compose( # include augmentation
        #     [
        #         ApplyTransformToKey(
        #             key="video",
        #             transform=Compose(
        #                 [
        #                     UniformTemporalSubsample(num_frames_to_sample),
        #                     Lambda(lambda x: x / 255.0),
        #                     Normalize(mean, std),
        #                     # RandomShortSideScale(min_size=256, max_size=320),
        #                     # RandomCrop(resize_to),
        #                     RandomShortSideScale(min_size=225, max_size=280),
        #                     CenterCrop(resize_to),
        #                     RandomHorizontalFlip(p=0.5),
        #                 ]
        #             ),
        #         ),
        #     ]
        # )

        self.transform = Compose(
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

        self.video_path_handler = VideoPathHandler()

        # Init clip sampler
        fps = 30
        clip_duration = num_frames_to_sample * sample_rate / fps
        self.clip_duration = clip_duration
        self.clip_sampler = pytorchvideo.data.make_clip_sampler("uniform", clip_duration)
    
    def load_video(self, video_path:str, no_label=False):
        if not no_label:
            label = video_path.split("/")[-2]
            label_id = self.label2id[label]
        else:
            label_id=None
            label=None
        try:
            video = self.video_path_handler.video_from_path(
                video_path,
                decode_audio=False,
                decoder="pyav",
            )
        except Exception as e:
            print(f'\031[1;32m [Error]\033[0m {e}')
            exit()

        return video, {"label": label_id, "label_name": label}

    
    def extract(self, video_path: str, no_label:bool=False):
        """
        Extract one feature from the full video in video_path
        Args:
            video_path: path of the video clip
        """
        # self._video_sampler = video_sampler([video_path])

        video, info_dict = self.load_video(video_path, no_label)
        feat_list = []
        is_last_clip = False
        self._next_clip_start_time = 0
        clip_times = []
        while not is_last_clip:    
            (
                clip_start,
                clip_end,
                clip_index,
                aug_index,
                is_last_clip,
            ) = self.clip_sampler(
                self._next_clip_start_time, video.duration, info_dict
            )

            assert aug_index == 0
            video_clip = video.get_clip(clip_start, clip_end)
            
            inputs = self.transform({"video": video_clip['video']})
            inputs = {"pixel_values": inputs["video"].permute(1, 0, 2, 3)[None, ...]}

            # inputs = self.collate_fn(inputs)

            outputs = self.model.videomae(**inputs)
            feat = outputs['last_hidden_state'].detach().cpu().numpy()[0,0]
            feat_list.append(feat)
            clip_times.append((float(clip_start/video.duration), float(clip_end/video.duration)))

            # NOTE: we add 25% overlapping
            self._next_clip_start_time = (clip_end - self.clip_duration/4)

        # Reset states
        video.close()
        self.clip_sampler.reset()
        info_dict.update({"video_duration": float(video.duration), "clip_times": clip_times})
        
        return feat_list, info_dict
    
    def get_clips(self, video_path, visualization=False):
        if visualization:
            transform = Compose(
                [
                    ApplyTransformToKey(
                        key="video",
                        transform=Compose(
                            [
                                UniformTemporalSubsample(self.model.config.num_frames*3),
                                Resize(320)
                            ]
                        ),
                    ),
                ]
            )
        else:
            transform=Compose(
                [
                    ApplyTransformToKey(key="video",transform=Compose([])),
                ]
            )

        video, info_dict = self.load_video(video_path, no_label=True)
        clip_list = []
        is_last_clip=False
        self._next_clip_start_time=0
        while not is_last_clip:    
            (
                clip_start,
                clip_end,
                clip_index,
                aug_index,
                is_last_clip,
            ) = self.clip_sampler(
                self._next_clip_start_time, video.duration, info_dict
            )

            video_clip = video.get_clip(clip_start, clip_end)
            frames = transform({"video": video_clip['video']})['video'].permute(1, 0, 2, 3)
            clip_list.append(frames.cpu().numpy())

            # NOTE: we add 25% overlapping
            self._next_clip_start_time = (clip_end - self.clip_duration/4)

        # Reset states
        video.close()
        self.clip_sampler.reset()
        return clip_list
    