import numpy as np
import moviepy.editor as mpy
import argparse
import pathlib
from utils.visualizer import Visualizer
import os
from sklearn.metrics.pairwise import cosine_similarity
from scipy.special import softmax

parser = argparse.ArgumentParser(description='arguments')
parser.add_argument('--data_root', type=str, default='./assets/dset-feats-sr5/', help='')
parser.add_argument('--test_root', type=str, default='./assets/dset-feat-test-sr5/', help='')
parser.add_argument('--cam_type', type=str, default='cam1', help='')
parser.add_argument('--mode', type=str, default='few_shot', choices=['few_shot', 'linear_prob'], help='')
parser.add_argument('--log_scale', action='store_true', default=False)
args = parser.parse_args()

label_ordered = [0, 5, 4 ,7, 1, 6, 2, 3]
def load_train_val(input_root):
    """
    Return support matrix [n_cls, feat_dim]
    """
    print('\033[1;32m [INFO]\033[0m Read support set (train+val) feats')
    train_feats=np.load(os.path.join(input_root, "train.npy"), allow_pickle=True).item()
    train_hidden_states = train_feats["feat"]
    train_subsamples = train_feats["feat_subsamples"]
    train_labels =  train_feats["labels"]

    val_feats=np.load(os.path.join(input_root, "val.npy"), allow_pickle=True).item()
    val_hidden_states = val_feats["feat"]
    val_subsamples = val_feats["feat_subsamples"]
    val_labels =  val_feats["labels"]

    return train_hidden_states, train_labels, val_hidden_states, val_labels

def load_test_feats(test_root, cam_type, test_ids= ['012', '014', '015', '016']):
    # Read test feats
    if cam_type=='cam1':
        test_path = os.path.join(test_root, '{test_id}', 'Camera1.npy')
    elif cam_type=='cam2':
        test_path = os.path.join(test_root, '{test_id}', 'Camera2.npy')
    else:
        raise NotImplementedError()
    
    hidden_dict = dict()
    labels = dict()
    id2label = dict()
    for test_id in test_ids:
        feats=np.load(test_path.format(test_id=test_id), allow_pickle=True).item()
        hidden_states = feats["feat"]
        hidden_dict[test_id] = hidden_states
        if feats['labels'] is not None:
            labels[test_id] = feats['labels']
        else:
            labels[test_id] = ["No records"]
        
        id2label = feats['id2label']

    return hidden_dict, labels, id2label

def few_shot_prototype(train_hidden_states, train_labels, val_hidden_states, val_labels, test_dict, log_scale=False):
    
    support_feats = np.concatenate([train_hidden_states, val_hidden_states], axis=0)
    support_labels = np.concatenate([train_labels, val_labels], axis=0)

    X = []
    for la in label_ordered:
        ins = np.where(support_labels == la)
        feats = support_feats[ins]
        X.append(np.mean(feats, axis=0))
    
    test_results = dict()
    for k,v in test_dict.items():
        test_feats = v
        mat = cosine_similarity(X, test_feats) # input [ncls, feat_dim] [nsample, feat_dim] output [ncls, nsample]
        mat = (mat +1)/2
        print(np.amin(mat), np.amax(mat))
        if log_scale:
            mat = np.power(mat, 10)

        mat = softmax(mat, axis=0)
        preds = np.argmax(mat, axis=0)
        test_results[k] = {"mat": mat, "preds": preds}
    return test_results


def main():
    cam_type = args.cam_type
    data_root = os.path.join(args.data_root, cam_type)
    train_hidden_states, train_labels, val_hidden_states, val_labels = load_train_val(data_root)
    
    test_dict, labels, id2label = load_test_feats(args.test_root, cam_type)
    if args.mode == 'few_shot':
        test_results = few_shot_prototype(
            train_hidden_states, train_labels, val_hidden_states, val_labels, test_dict,
            log_scale=args.log_scale)
    elif args.mode == 'linear_prob':
        test_results = np.load(os.path.join(args.test_root, 'linearprob_test.npy'), allow_pickle=True).item()
        for k,v in test_results.items():
            v['mat'] = v['mat'][label_ordered]
            v['preds'] = [label_ordered.index(pred) for pred in v['preds']]
    else:
        raise NotImplementedError()

    print('\033[1;32m [INFO]\033[0m Start visualization')
    key_mapping = {"cam1": "Camera1", "cam2": "Camera2"}
    for k,v in test_results.items():
        print(f'\033[1;32m [INFO]\033[0m Running visualization of video index {k}')
        mat = v['mat']
        pred_results = [id2label[label_ordered[i]] for i in v['preds']]

        sub_folder = os.path.join(args.test_root, k, key_mapping[cam_type])
        
        vis = Visualizer(mat=mat, sub_folder=sub_folder, meta={"vid": k, "gt_seq":labels[k], "pred_seq": pred_results})
        vis.run(port=8091)


if __name__ == '__main__':
    main()

