"""Optimization module"""
import needle as ndl
import numpy as np


class Optimizer:
    def __init__(self, params):
        self.params = params

    def step(self):
        raise NotImplementedError()

    def reset_grad(self):
        for p in self.params:
            p.grad = None


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.u = {}
        self.weight_decay = weight_decay

    def step(self):
        ### BEGIN YOUR SOLUTION
        for param in self.params:
            grad = self.momentum * self.u.get(param, 0) + (1 - self.momentum) * (param.grad.data + self.weight_decay * param.data)
            self.u[param] = ndl.Tensor(grad, dtype=param.dtype)
            param.data -= self.lr * self.u[param]
        ### END YOUR SOLUTION

    def clip_grad_norm(self, max_norm=0.25):
        """
        Clips gradient norm of parameters.
        """
        ### BEGIN YOUR SOLUTION
        total_norm = np.linalg.norm(np.array([np.linalg.norm(p.grad.detach().numpy()).reshape((1,)) for p in self.params]))
        clip_coef = max_norm / (total_norm + 1e-6)
        clip_coef_clamped = min((np.asscalar(clip_coef), 1.0))
        for p in self.params:
            p.grad = p.grad.detach() * clip_coef_clamped
        ### END YOUR SOLUTION


class Adam(Optimizer):
    def __init__(
        self,
        params,
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
    ):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

        self.m = {}
        self.v = {}

    def step(self):
        ### BEGIN YOUR SOLUTION
        self.t += 1
        for param in self.params:
            deltaf = param.grad.data + self.weight_decay * param.data
            u_t = self.beta1 * self.m.get(param, 0) + (1 - self.beta1) * deltaf
            # u_t = ndl.Tensor(u_t, dtype=param.dtype)
            self.m[param] = u_t
            v_t = self.beta2 * self.v.get(param, 0) + (1 - self.beta2) * (deltaf ** 2)
            # v_t = ndl.Tensor(v_t, dtype=param.dtype)
            self.v[param] = v_t
            
            unbiased_u = self.m[param] / (1 - self.beta1 ** self.t)
            unbiased_v = self.v[param] / (1 - self.beta2 ** self.t)
            update = self.lr * unbiased_u.data / (unbiased_v.data ** 0.5 + self.eps)
            update = ndl.Tensor(update, dtype=param.dtype)
            # print(update)
            param.data -= update.data
        ### END YOUR SOLUTION
