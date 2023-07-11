# python step1_preprocess_videos.py --frame_shift 5

# python step2_random_split_dataset.py

python step3_feat_extraction.py --sample_rate 5 --out_root "../data/General/processed_split/dset-feats-sr5"

# python prepare_test_vis.py --sample_rate 5 --out_root "../data/General/processed_split/dset-feat-test-sr5" --for_vis