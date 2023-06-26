import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os
import pprint

input_root="outputs/feat_ssv2_finetune"
input_root="outputs/feat_kinetics_finetune"

# Read train+val feats
print('\033[1;32m [INFO]\033[0m Read support set (train+val) feats')
train_feats=np.load(os.path.join(input_root, "train.npy"), allow_pickle=True).item()
train_hidden_states = train_feats["hidden"]
train_labels =  train_feats["labels"]

val_feats=np.load(os.path.join(input_root, "val.npy"), allow_pickle=True).item()
val_hidden_states = val_feats["hidden"]
val_labels =  val_feats["labels"]

support_feats = np.concatenate([train_hidden_states, val_hidden_states], axis=0)
support_labels = np.concatenate([train_labels, val_labels], axis=0)

id2labels = train_feats["id2label"]
labels_unique = np.unique(support_labels)

X = []
for la in labels_unique:
    ins = np.where(support_labels == la)
    feats = support_feats[ins]
    X.append(np.mean(feats, axis=0))


# Read test feats
print('\033[1;32m [INFO]\033[0m Read test set (test) feats')
train_feats=np.load(os.path.join(input_root, "test.npy"), allow_pickle=True).item()
hidden_states = train_feats["hidden"]
labels =  train_feats["labels"]
id2labels = train_feats["id2label"]
labels_unique = np.unique(labels)


print('\033[1;32m [INFO]\033[0m Run prototype matching')
results = {}
for la in labels_unique:
    ins = np.where(labels == la)
    feats = hidden_states[ins]
    mat = cosine_similarity(X, feats) # input [ncls, feat_dim] [nsample, feat_dim] output [ncls, nsample]
    preds = np.argmax(mat, axis=0)
    truth_table = labels[ins] == preds
    acc = sum(truth_table)/len(truth_table)
    results[id2labels[la]] = {"acc": acc, "num_correct": sum(truth_table), "tot": len(truth_table)}

pprint.pprint(results)

cnt = 0
tot=0
for k,v in results.items():
    cnt += v["num_correct"]
    tot += v["tot"]
print(f"Overall test accuracy: {cnt/tot} ({cnt}/{tot})")