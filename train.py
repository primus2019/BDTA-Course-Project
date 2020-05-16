import time
import argparse
import numpy as np
import random

import torch
import torch.nn.functional as F

from utils import load_data, load_rand_split_data, accuracy
from model import GCN

# hyper-params
dataset = 'citeseer' # Citeseer or cora
seed = 24 # Random seed
hidden = 16 # Number of hidden units
dropout = 0.5 # Dropout rate
lr = 0.01 # Learning rate
weight_decay = 5e-4 # Weight decay(L2 loss)
epochs = 200 # Train epochs


def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features, adj)
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()
    model.eval()
    output = model(features, adj)

    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}s'.format(time.time() - t))


def test():
    model.eval()
    output = model(features, adj)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))
    return loss_test.item(), acc_test.item()


if __name__ == '__main__':
    random.seed(seed)
    rg = np.random.default_rng(seed)
    torch.manual_seed(seed)

    # Load data
    adj, features, labels, idx_train, idx_val, idx_test = load_data(dataset=dataset, seed=seed)
    # adj, features, labels, idx_train, idx_val, idx_test = load_rand_loadsplit_data(dataset)

    # Model and optimizer
    model = GCN(nfeat=features.shape[1],
                nhid=hidden,
                nclass=labels.max().item() + 1,
                dropout=dropout)
    optimizer = torch.optim.Adam(model.parameters(),
                        lr=lr, weight_decay=weight_decay)

    # Train model
    t_total = time.time()
    for epoch in range(epochs):
        train(epoch)
    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

    # Testing
    res = test()