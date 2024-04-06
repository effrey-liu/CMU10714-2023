import sys
sys.path.append('./python')
from needle.autograd import Tensor
import needle as ndl
import needle.nn as nn
import math
import numpy as np
np.random.seed(0)


# class ConvBN(ndl.nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, stride, device=None, dtype="float32"):
#         super().__init__()
#         self.conv = nn.Conv(in_channels, out_channels, kernel_size, stride=stride, device=device, dtype=dtype)
#         self.bn = nn.BatchNorm2d(out_channels, device=device, dtype=dtype)
#         self.relu = nn.ReLU()
        
#     def forward(self, x: Tensor) -> Tensor:
#         x = self.conv(x)
#         x = self.bn(x)
#         x = self.relu(x)
#         return x
def ConvBN(in_channels, out_channels, kernel_size, stride, device=None, dtype="float32"):
    return nn.Sequential(
        nn.Conv(in_channels, out_channels, kernel_size=kernel_size, stride=stride, bias=True, device=device, dtype=dtype),
        nn.BatchNorm2d(out_channels, device=device, dtype=dtype),
        nn.ReLU()
    )

class ResidualBlock(ndl.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, device=None, dtype="float32"):
        super().__init__()
        self.conv1 = ConvBN(in_channels, out_channels, kernel_size, stride, device=device, dtype=dtype)
        self.conv2 = ConvBN(out_channels, out_channels, kernel_size, stride, device=device, dtype=dtype)
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        return out + x

class ResNet9(ndl.nn.Module):
    def __init__(self, device=None, dtype="float32"):
        super().__init__()
        ### BEGIN YOUR SOLUTION
        self.conv1 = ConvBN(3, 16, kernel_size=7, stride=4, device=device, dtype=dtype)
        self.conv2 = ConvBN(16, 32, kernel_size=3, stride=2, device=device, dtype=dtype)
        self.res_block1 = ResidualBlock(32, 32, kernel_size=3, stride=1, device=device, dtype=dtype)
        self.conv3 = ConvBN(32, 64, kernel_size=3, stride=2, device=device, dtype=dtype)
        self.conv4 = ConvBN(64, 128, kernel_size=3, stride=2, device=device, dtype=dtype)
        self.res_block2 = ResidualBlock(128, 128, kernel_size=3, stride=1, device=device, dtype=dtype)
        self.linear1 = nn.Linear(128, 128, device=device, dtype=dtype)
        self.linear2 = nn.Linear(128, 10, device=device, dtype=dtype)
        ### END YOUR SOLUTION

    def forward(self, x):
        ### BEGIN YOUR SOLUTION
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.res_block1(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res_block2(out)
        out = nn.Flatten()(out)
        out = self.linear1(out)
        out = ndl.ops.relu(out)
        out = self.linear2(out)
        return out
        ### END YOUR SOLUTION


class LanguageModel(nn.Module):
    def __init__(self, embedding_size, output_size, hidden_size, num_layers=1,
                 seq_model='rnn', device=None, dtype="float32"):
        """
        Consists of an embedding layer, a sequence model (either RNN or LSTM), and a
        linear layer.
        Parameters:
        output_size: Size of dictionary
        embedding_size: Size of embeddings
        hidden_size: The number of features in the hidden state of LSTM or RNN
        seq_model: 'rnn' or 'lstm', whether to use RNN or LSTM
        num_layers: Number of layers in RNN or LSTM
        """
        super(LanguageModel, self).__init__()
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def forward(self, x, h=None):
        """
        Given sequence (and the previous hidden state if given), returns probabilities of next word
        (along with the last hidden state from the sequence model).
        Inputs:
        x of shape (seq_len, bs)
        h of shape (num_layers, bs, hidden_size) if using RNN,
            else h is tuple of (h0, c0), each of shape (num_layers, bs, hidden_size)
        Returns (out, h)
        out of shape (seq_len*bs, output_size)
        h of shape (num_layers, bs, hidden_size) if using RNN,
            else h is tuple of (h0, c0), each of shape (num_layers, bs, hidden_size)
        """
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


if __name__ == "__main__":
    model = ResNet9()
    x = ndl.ops.randu((1, 32, 32, 3), requires_grad=True)
    model(x)
    cifar10_train_dataset = ndl.data.CIFAR10Dataset("data/cifar-10-batches-py", train=True)
    train_loader = ndl.data.DataLoader(cifar10_train_dataset, 128, ndl.cpu(), dtype="float32")
    print(dataset[1][0].shape)