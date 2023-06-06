# videomae_srris

data files:
```
../data/General/Annotations
../data/General/Videos
```

split video to clips and preprocess data to Ucf101 format

```bash
python preprocess_videos.py
python random_split_dataset.py
```

train and evaluate model
```bash
python main.py
```
