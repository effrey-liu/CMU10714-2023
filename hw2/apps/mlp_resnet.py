import sys

sys.path.append("../python")
import needle as ndl
import needle.nn as nn
import numpy as np
import time
import os
from needle.data import MNISTDataset, DataLoader

np.random.seed(0)
# MY_DEVICE = ndl.backend_selection.cuda()


def ResidualBlock(dim, hidden_dim, norm=nn.BatchNorm1d, drop_prob=0.1):
    ### BEGIN YOUR SOLUTION
    main_path = nn.Sequential(nn.Linear(dim, hidden_dim), norm(hidden_dim), nn.ReLU(),
                              nn.Dropout(drop_prob), nn.Linear(hidden_dim, dim), norm(dim))
    res = nn.Residual(main_path)
    return nn.Sequential(res, nn.ReLU())
    ### END YOUR SOLUTION


def MLPResNet(
    dim,
    hidden_dim=100,
    num_blocks=3,
    num_classes=10,
    norm=nn.BatchNorm1d,
    drop_prob=0.1,
):
    ### BEGIN YOUR SOLUTION
    return nn.Sequential(nn.Linear(dim, hidden_dim), nn.ReLU(),
                         *[ResidualBlock(dim=hidden_dim, hidden_dim=hidden_dim // 2, norm=norm, drop_prob=drop_prob) for _ in range(num_blocks)],
                         nn.Linear(hidden_dim, num_classes))
    ### END YOUR SOLUTION


def epoch(dataloader, model, opt=None):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    tot_loss, tot_error = [], 0.0
    loss_fn = nn.SoftmaxLoss()
    if opt is None:
        model.eval()
        for x, y in dataloader:
            # print(type(x), x.shape)
            logits = model(x)
            loss = loss_fn(logits, y)
            tot_error += np.sum(logits.numpy().argmax(axis = 1) != y.numpy())
            tot_loss.append(loss.numpy())
    else:
        model.train()
        for x, y in dataloader:
            # print(type(x), x.shape)
            logits = model(x)
            loss = loss_fn(logits, y)
            tot_error += np.sum(logits.numpy().argmax(axis = 1) != y.numpy())
            tot_loss.append(loss.numpy())
            opt.reset_grad()
            loss.backward()
            opt.step()
    sample_num = len(dataloader.dataset)
    return tot_error / sample_num, np.mean(tot_loss)
    ### END YOUR SOLUTION


def train_mnist(
    batch_size=100,
    epochs=10,
    optimizer=ndl.optim.Adam,
    lr=0.001,
    weight_decay=0.001,
    hidden_dim=100,
    data_dir="data",
):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    resnet = MLPResNet(28*28, hidden_dim=hidden_dim, num_classes=10)
    opt = optimizer(resnet.parameters(), lr=lr, weight_decay=weight_decay)
    train_set = MNISTDataset(f"{data_dir}/train-images-idx3-ubyte.gz", 
                             f"{data_dir}/train-labels-idx1-ubyte.gz")
    test_set = MNISTDataset(f"{data_dir}/t10k-images-idx3-ubyte.gz",
                            f"{data_dir}/t10k-labels-idx1-ubyte.gz")
    train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle = True)
    test_dataloader = DataLoader(test_set, batch_size=batch_size)
    
    for _ in range(epochs):
        train_err, train_loss = epoch(dataloader=train_dataloader, model=resnet, opt=opt)
        print("train_err: %d, train_loss: %d" % (train_err, train_loss))
    test_err, test_loss = epoch(dataloader=test_dataloader, model=resnet, opt=None)
    
    return train_err, train_loss, test_err, test_loss
    ### END YOUR SOLUTION


if __name__ == "__main__":
    # train_mnist(data_dir="../data")
    np.random.seed(1)
    train_err, train_loss, test_err, test_loss = train_mnist(batch_size=250, epochs=5, optimizer=ndl.optim.SGD, lr=0.01, weight_decay=0.001, hidden_dim=100, data_dir="../data")
    print("Eventually, train_err: %d, train_loss: %d, test_err: %d, test_loss: %d" % (train_err, train_loss, test_err, test_loss))
