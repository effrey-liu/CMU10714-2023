"""Operator implementations."""

from numbers import Number
from typing import Optional, List, Tuple, Union

from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp
import numpy

# NOTE: we will import numpy as the array_api
# as the backend for our computations, this line will change in later homeworks

import numpy as array_api


class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        return array_api.power(a, self.scalar)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        input, = node.inputs
        return out_grad * self.scalar * array_api.power(input, self.scalar - 1)
        ### END YOUR SOLUTION


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWisePow(TensorOp):
    """Op to element-wise raise a tensor to a power."""

    def compute(self, a: NDArray, b: NDArray) -> NDArray:
        return a**b

    def gradient(self, out_grad, node):
        if not isinstance(node.inputs[0], NDArray) or not isinstance(
            node.inputs[1], NDArray
        ):
            raise ValueError("Both inputs must be tensors (NDArray).")

        a, b = node.inputs[0], node.inputs[1]
        grad_a = out_grad * b * (a ** (b - 1))
        grad_b = out_grad * (a**b) * array_api.log(a.data)
        return grad_a, grad_b

def power(a, b):
    return EWisePow()(a, b)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return a / b
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # if not isinstance(node.inputs[0], NDArray) or not isinstance(
        #     node.inputs[1], NDArray
        # ):
        #     raise ValueError("Both inputs must be tensors (NDArray).")
        
        a, b = node.inputs[0], node.inputs[1]
        grad_a = out_grad / b
        grad_b = out_grad * (-a) / (b * b)
        return grad_a, grad_b
        ### END YOUR SOLUTION


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a / self.scalar
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad / self.scalar
        ### END YOUR SOLUTION


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        if self.axes:
            x, y = self.axes[0], self.axes[1]
        else:
            x, y = -1, -2
        return array_api.swapaxes(a, x, y)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        if self.axes:
            x, y = self.axes[0], self.axes[1]
        else:
            x, y = -1, -2
        # print("x = %d, y = %d" % (x, y))
        # print(type(out_grad), out_grad.shape, out_grad)
        # transpose_result = transpose(out_grad, axes=(x, y))
        # print(type(transpose_result), transpose_result.shape, transpose_result)
        # swap_result = array_api.swapaxes(out_grad, x, y)      # numpy.AxisError: axis1: axis 1 is out of bounds for array of dimension 0
        # print(type(swap_result), swap_result.shape, swap_result)
        
        return transpose(out_grad, axes=(x, y))     # Correct!
        # return array_api.swapaxes(out_grad, x, y) # Wrong???
        """
        猜测原因是：
        array_api.swapaxes(out_grad, x, y)没有继续使用计算图，
        只得到了一个结果，但是反向传播的时候，要生成计算图。
        任务描述里：
        We will discuss the `gradient()` call in the next section, 
        but it is important to emphasize here that 
        this call is different from forward in that it takes `Tensor` arguments.
        This means that any call you make within this function should be done via `TensorOp` operations themselves 
        (so that you can take gradients of gradients).
        """
        ### END YOUR SOLUTION


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.reshape(a, self.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        input, = node.inputs
        return reshape(out_grad, input.shape)
        ### END YOUR SOLUTION


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.broadcast_to(a, self.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        input, = node.inputs
        input_shape = input.shape
        output_shape = self.shape
        
        expand_dim = len(output_shape) - len(input_shape)
        reduced_dims = []
        for i in range(expand_dim):     # 先将多余的维度进行reduce，也就是sum
            out_grad = summation(out_grad, axes=(0))
        for i in range(len(input_shape)):   # 然后检查是否在某一维度上进行了广播
            if input_shape[i] != output_shape[i + expand_dim]:
                # 该维度上大小不一致说明进行了广播，记录下来，将该维度上进行sum
                reduced_dims.append(i)
        return reshape(summation(out_grad, axes=tuple(reduced_dims)), input_shape)
        ### END YOUR SOLUTION


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):      # 和Broadcast_to类似，但summation当中需要找到哪些维度被消减了
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.sum(a, axis=self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        input, = node.inputs
        input_shape = input.shape
        output_shape = out_grad.shape
        
        expand_dims = list(input_shape)    # 需要扩展到什么维度
        if self.axes is None:       # 说明summation的结果是矩阵里所有值的和
            axes = list(range(len(input_shape)))
        else:                       # 说明规定了在哪些维度上求和
            axes = self.axes
        for i in range(len(axes)):
            expand_dims[axes[i]] = 1
        out_grad = reshape(out_grad, expand_dims)   # 先把缺少的维度恢复
        return broadcast_to(out_grad, input_shape)  # 进行广播
        ### END YOUR SOLUTION


def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return array_api.matmul(a, b)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        lhs, rhs = node.inputs
        dlhs = matmul(out_grad, transpose(rhs))
        drhs = matmul(transpose(lhs), out_grad)
        
        if dlhs.shape != lhs.shape:
            dlhs = summation(dlhs, tuple(range(len(dlhs.shape) - len(lhs.shape))))
        if drhs.shape != rhs.shape:
            drhs = summation(drhs, tuple(range(len(drhs.shape) - len(rhs.shape))))
        
        return dlhs, drhs
        ### END YOUR SOLUTION


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.negative(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return negate(out_grad)
        ### END YOUR SOLUTION


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.log(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        input, = node.inputs
        return out_grad / input
        ### END YOUR SOLUTION


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.exp(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        input, = node.inputs
        return out_grad * exp(input)
        ### END YOUR SOLUTION


def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.maximum(a, 0)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        input, = node.inputs
        # print(type(input), input.shape, input)
        # mask = input > 0      # TypeError: '>' not supported between instances of 'Tensor' and 'int'
        mask = input.numpy() > 0
        return out_grad * mask
        ### END YOUR SOLUTION


def relu(a):
    return ReLU()(a)
