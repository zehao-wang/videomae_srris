import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import os
from collections import defaultdict
from scipy.special import softmax

input_root='./assets/dset-feats-sr5/cam1'
test_root='./assets/dset-feat-test-sr5/'
epochs = 100

model = nn.Sequential(
        nn.Linear(768, 64),
        nn.ReLU(),
        nn.Linear(64, 8)
).cuda()

optimizer = torch.optim.Adam(model.parameters())

class FeatDataset(Dataset):
    def __init__(self, input_path, mode='train'):
        feats=np.load(input_path, allow_pickle=True).item()
        self.id2labels =  feats["id2label"]

        if mode == 'train':
            self.hidden_states = []
            self.labels = []
            for label, subsample in zip(feats["labels"], feats["feat_subsamples"]):
                self.hidden_states += subsample
                self.labels += len(subsample) * [label]
        else:
            self.hidden_states = feats['feat']
            self.labels = feats['labels']

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        datum = self.hidden_states[idx]
        label = self.labels[idx]
        return datum, label

train_dset = FeatDataset(os.path.join(input_root, "train.npy"))
val_dset = FeatDataset(os.path.join(input_root, "val.npy"), mode='val')
test_dset = FeatDataset(os.path.join(input_root, "test.npy"), mode='test')

train_dataloader = DataLoader(train_dset, batch_size=64, shuffle=True)
val_dataloader = DataLoader(val_dset, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_dset, batch_size=64, shuffle=True)

def train(dataloader, model, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.cuda(), y.cuda()

        # Compute prediction error
        pred = model(X)
        loss = nn.functional.cross_entropy(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        loss, current = loss.item(), (batch + 1) * len(X)
        # print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def val(dataloader, model, epoch):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.cuda(), y.cuda()
            pred = model(X)
            test_loss += nn.functional.cross_entropy(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Epoch {epoch} Validation Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return 100*correct

def test(dataloader, model, epoch):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    per_cls_results = defaultdict(list)
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.cuda(), y.cuda()
            pred = model(X)
            test_loss += nn.functional.cross_entropy(pred, y).item()

            for is_correct, gt_label in zip(pred.argmax(1)==y, y):
                per_cls_results[dataloader.dataset.id2labels[gt_label.item()]].append(is_correct.item())

            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Epoch {epoch} Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    print("Per cls accuracy: ")
    for k in sorted(list(per_cls_results.keys())):
        v = per_cls_results[k]
        print(f"{k}:\t{np.mean(v)} ({np.sum(v)} / {len(v)})")
    return 100* correct

cached_best = {"acc": 0.0, "acc_test": 0.0}
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, optimizer)
    acc_val = val(val_dataloader, model, t+1)
    acc_test =test(test_dataloader, model, t+1)
    if acc_val > cached_best['acc']:
        cached_best['acc'] = acc_val
        cached_best['acc_test'] = acc_test
    print(f"Best: val {cached_best['acc']}, test {cached_best['acc_test']}")
print("Done!")


# NOTE: extract test feat from std test data
print('\033[1;32m [INFO]\033[0m Extracting test video clip prob')
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

cam_type = input_root.split('/')[-1]
hidden_dict, labels, id2label = load_test_feats(test_root, cam_type)

test_results = {}
for test_id,v in hidden_dict.items():
    inputs = torch.from_numpy(np.stack(v)).cuda()
    model.eval()
    with torch.no_grad():
        pred = model(inputs)
        mat = softmax(pred.cpu().numpy(), axis=1)
        preds = np.argmax(mat, axis=1)

        test_results[test_id] = {"mat": mat.T , "preds": preds}

np.save(os.path.join(test_root, "linearprob_test.npy"), test_results)