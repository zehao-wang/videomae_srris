import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import os
input_root='./outputs/feat_ssv2_finetune'
epochs = 100

model = nn.Sequential(
        nn.Linear(768, 8),
).cuda()

optimizer = torch.optim.Adam(model.parameters())


class FeatDataset(Dataset):
    def __init__(self, input_path):
        train_feats=np.load(input_path, allow_pickle=True).item()
        self.train_hidden_states = train_feats["hidden"]
        self.train_labels =  train_feats["labels"]

    def __len__(self):
        return len(self.train_labels)

    def __getitem__(self, idx):
        datum = self.train_hidden_states[idx]
        label = self.train_labels[idx]
        return datum, label

train_dset = FeatDataset(os.path.join(input_root, "train.npy"))
val_dset = FeatDataset(os.path.join(input_root, "val.npy"))
test_dset = FeatDataset(os.path.join(input_root, "test.npy"))

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
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.cuda(), y.cuda()
            pred = model(X)
            test_loss += nn.functional.cross_entropy(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Epoch {epoch} Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
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