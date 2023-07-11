import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os
import pprint

KNN = 8
KNN_MODE=False

input_root='../data/General/processed_split/dset-feats-sr5/cam1'

# Read train+val feats
print('\033[1;32m [INFO]\033[0m Read support set (train+val) feats')
train_feats=np.load(os.path.join(input_root, "train.npy"), allow_pickle=True).item()
train_hidden_states = train_feats["feat"]
train_subsamples = train_feats["feat_subsamples"]
train_labels =  train_feats["labels"]

val_feats=np.load(os.path.join(input_root, "val.npy"), allow_pickle=True).item()
val_hidden_states = val_feats["feat"]
val_subsamples = val_feats["feat_subsamples"]
val_labels =  val_feats["labels"]
id2labels = train_feats["id2label"]

if KNN_MODE:
    assert len(train_labels+val_labels) == len(train_subsamples+val_subsamples)
    support_feats = []
    support_labels = []
    for label, subsample in zip(train_labels+val_labels, train_subsamples+val_subsamples):
        support_feats += subsample
        support_labels += len(subsample) * [label]
    support_feats = np.array(support_feats)
    support_labels = np.array(support_labels)
    X = support_feats

else:
    support_feats = np.concatenate([train_hidden_states, val_hidden_states], axis=0)
    support_labels = np.concatenate([train_labels, val_labels], axis=0)

    labels_unique = np.unique(support_labels)
    X = []
    for la in sorted(labels_unique):
        ins = np.where(support_labels == la)
        feats = support_feats[ins]
        X.append(np.mean(feats, axis=0))



# Read test feats
print('\033[1;32m [INFO]\033[0m Read test set (test) feats')
test_feats=np.load(os.path.join(input_root, "test.npy"), allow_pickle=True).item()
hidden_states = np.stack(test_feats["feat"])
test_subsamples = test_feats["feat_subsamples"]
labels =  np.stack(test_feats["labels"])
id2labels = test_feats["id2label"]
labels_unique = np.unique(labels)


print('\033[1;32m [INFO]\033[0m Run prototype matching')
results = {}
for la in labels_unique:
    ins = np.where(labels == la)
    feats = hidden_states[ins]
    mat = cosine_similarity(X, feats) # input [ncls, feat_dim] [nsample, feat_dim] output [ncls, nsample]
    if KNN_MODE:
        preds = []
        for i in range(mat.shape[1]):
            inds = np.argpartition(mat[:, i], -KNN)[-KNN:]
            selected_labels = support_labels[inds].tolist()
            preds.append(max(set(selected_labels), key=selected_labels.count))
        preds = np.array(preds)
    else:
        preds = np.argmax(mat, axis=0)

    truth_table = labels[ins] == preds
    acc = sum(truth_table)/len(truth_table)
    results[id2labels[la]] = {"acc": acc, "num_correct": sum(truth_table), "tot": len(truth_table)}

for k in sorted(list(results.keys())):
    v = results[k]
    print(f"{k}:\t{v['acc']} ({v['num_correct']} / {v['tot']})")

cnt = 0
tot=0
for k,v in results.items():
    cnt += v["num_correct"]
    tot += v["tot"]
print(f"Overall test accuracy: {cnt/tot} ({cnt}/{tot})")