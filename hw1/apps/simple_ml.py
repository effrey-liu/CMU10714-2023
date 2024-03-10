"""hw1/apps/simple_ml.py"""

import struct
import gzip
import numpy as np

import sys

sys.path.append("python/")
import needle as ndl


def parse_mnist(image_filesname, label_filename):
    """Read an images and labels file in MNIST format.  See this page:
    http://yann.lecun.com/exdb/mnist/ for a description of the file format.

    Args:
        image_filename (str): name of gzipped images file in MNIST format
        label_filename (str): name of gzipped labels file in MNIST format

    Returns:
        Tuple (X,y):
            X (numpy.ndarray[np.float32]): 2D numpy array containing the loaded
                data.  The dimensionality of the data should be
                (num_examples x input_dim) where 'input_dim' is the full
                dimension of the data, e.g., since MNIST images are 28x28, it
                will be 784.  Values should be of type np.float32, and the data
                should be normalized to have a minimum value of 0.0 and a
                maximum value of 1.0.

            y (numpy.ndarray[dypte=np.int8]): 1D numpy array containing the
                labels of the examples.  Values should be of type np.int8 and
                for MNIST will contain the values 0-9.
    """
    ### BEGIN YOUR SOLUTION
    with gzip.open(image_filesname, 'rb') as f:
        image = f.read()
        X = np.frombuffer(image, dtype=np.uint8, offset=16).astype(np.float32)
        X = X / 255
        # print(type(X))
        X = np.reshape(X, (-1, 784))

    
    with gzip.open(label_filename, "rb") as f:
        label = f.read()
        y = np.frombuffer(label, dtype=np.uint8, offset=8).astype(np.uint8)
        # print(type(y))
        
    return X, y
    ### END YOUR SOLUTION


def softmax_loss(Z, y_one_hot):
    """Return softmax loss.  Note that for the purposes of this assignment,
    you don't need to worry about "nicely" scaling the numerical properties
    of the log-sum-exp computation, but can just compute this directly.

    Args:
        Z (ndl.Tensor[np.float32]): 2D Tensor of shape
            (batch_size, num_classes), containing the logit predictions for
            each class.
        y (ndl.Tensor[np.int8]): 2D Tensor of shape (batch_size, num_classes)
            containing a 1 at the index of the true label of each example and
            zeros elsewhere.

    Returns:
        Average softmax loss over the sample. (ndl.Tensor[np.float32])
    """
    ### BEGIN YOUR SOLUTION
    """
    # predication = np.exp(Z) / np.sum(np.exp(Z), axis=1, keepdims=True)
    # return np.mean(-np.log(predication[np.indices(y.shape)[0], y]))
    # 这里不再是自己生成y_one_hot, 而是运用另外一个公式: lecture2 - softmax regression P12, P13
    """
    batch_size = Z.shape[0]
    nomorlization = ndl.log(ndl.summation(ndl.exp(Z), axes = 1))
    loss = ndl.summation(nomorlization - ndl.summation(y_one_hot * Z, axes = 1))
    return loss / batch_size
    ### END YOUR SOLUTION


def nn_epoch(X, y, W1, W2, lr=0.1, batch=100):
    """Run a single epoch of SGD for a two-layer neural network defined by the
    weights W1 and W2 (with no bias terms):
        logits = ReLU(X * W1) * W1
    The function should use the step size lr, and the specified batch size (and
    again, without randomizing the order of X).

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        W1 (ndl.Tensor[np.float32]): 2D array of first layer weights, of shape
            (input_dim, hidden_dim)
        W2 (ndl.Tensor[np.float32]): 2D array of second layer weights, of shape
            (hidden_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD mini-batch

    Returns:
        Tuple: (W1, W2)
            W1: ndl.Tensor[np.float32]
            W2: ndl.Tensor[np.float32]
    """

    ### BEGIN YOUR SOLUTION
    num_examples = X.shape[0]
    iter_num = num_examples // batch
    for iter in range(iter_num):
        #  AttributeError: 'numpy.ndarray' object has no attribute 'requires_grad'
        x_batch = X[iter * batch : (iter + 1) * batch, :]
        y_batch = y[iter * batch : (iter + 1) * batch]
        
        x_batch = ndl.Tensor(x_batch, dtype = "float32")
        
        Z_1 = ndl.matmul(x_batch, W1)
        Relu_Z1 = ndl.relu(Z_1)
        Z_2 = ndl.matmul(Relu_Z1, W2)
        
        e_y = np.zeros((batch, W2.shape[1]))
        e_y[range(len(y_batch)), y_batch] = 1
        
        e_y = ndl.Tensor(e_y, dtype = "float32")
        
        cross_entropy_loss = softmax_loss(Z_2, e_y)
        
        cross_entropy_loss.backward()
        
        # W1 -= lr * W1.grad
        # W2 -= lr * W2.grad
        W1 = ndl.Tensor(W1.realize_cached_data() - lr * W1.grad.realize_cached_data())
        W2 = ndl.Tensor(W2.realize_cached_data() - lr * W2.grad.realize_cached_data())
    return (W1, W2)
    ### END YOUR SOLUTION


### CODE BELOW IS FOR ILLUSTRATION, YOU DO NOT NEED TO EDIT


def loss_err(h, y):
    """Helper function to compute both loss and error"""
    y_one_hot = np.zeros((y.shape[0], h.shape[-1]))
    y_one_hot[np.arange(y.size), y] = 1
    y_ = ndl.Tensor(y_one_hot)
    return softmax_loss(h, y_).numpy(), np.mean(h.numpy().argmax(axis=1) != y)
