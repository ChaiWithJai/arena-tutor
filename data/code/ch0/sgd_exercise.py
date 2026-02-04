"""
Exercise: Implement SGD

This exercise asks you to implement Stochastic Gradient Descent with momentum.

The pseudocode you need to implement is:

    b_0 = 0  (already done in __init__)

    for t = 1 to ... do:
        g_t = grad(theta)              # This is already in param.grad
        if lambda != 0:
            g_t = g_t + lambda * theta  # weight decay
        if mu != 0:
            b_t = mu * b_{t-1} + g_t    # momentum update
            g_t = b_t
        theta = theta - lr * g_t        # parameter update

Where:
- theta: parameters
- g: gradients
- b: momentum buffer
- lambda (lmda): weight_decay
- mu: momentum
- lr (gamma): learning rate
"""

import sys
from pathlib import Path
from typing import Iterable

import torch as t

# Make sure exercises are in the path
chapter = "chapter0_fundamentals"
section = "part3_optimization"
root_dir = next(p for p in Path.cwd().parents if (p / chapter).exists())
exercises_dir = root_dir / chapter / "exercises"
if str(exercises_dir) not in sys.path:
    sys.path.append(str(exercises_dir))

import part3_optimization.tests as tests


class SGD:
    def __init__(
        self,
        params: Iterable[t.nn.parameter.Parameter],
        lr: float,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
    ):
        """Implements SGD with momentum.

        Like the PyTorch version, but assume nesterov=False, maximize=False, and dampening=0
            https://pytorch.org/docs/stable/generated/torch.optim.SGD.html#torch.optim.SGD
        """
        self.params = list(params)  # turn params into a list (generator would be consumed)
        self.lr = lr
        self.mu = momentum
        self.lmda = weight_decay

        # Initialize momentum buffer (b_0 = 0)
        self.b = [t.zeros_like(p) for p in self.params]

    def zero_grad(self) -> None:
        """Zeros all gradients of the parameters in `self.params`."""
        for param in self.params:
            param.grad = None

    @t.inference_mode()
    def step(self) -> None:
        for b, theta in zip(self.b, self.params):
            g = theta.grad
            if self.lmda != 0:
                g = g + self.lmda * theta
            if self.mu != 0:
                b.copy_(self.mu * b + g)
                g = b
            theta -= self.lr * g
        pass
        """Performs a single optimization step of the SGD algorithm.

        TODO: Implement this method following the pseudocode above.

        Hints:
        - Loop over zip(self.b, self.params) to get each momentum buffer and parameter
        - Get the gradient from param.grad (it's already computed by loss.backward())
        - If weight_decay (self.lmda) != 0, add it to the gradient: g = g + lmda * theta
        - If momentum (self.mu) != 0, update the momentum buffer and use it as g
        - Update the parameter in-place: theta -= lr * g

        Important:
        - Don't modify param.grad directly (create a new variable for g)
        - Do modify self.b[i] in-place using .copy_()
        - Do modify theta (param) in-place using -= or similar
        """
        # YOUR CODE HERE
        pass

    def __repr__(self) -> str:
        return f"SGD(lr={self.lr}, momentum={self.mu}, weight_decay={self.lmda})"


if __name__ == "__main__":
    print("Running SGD tests...")
    tests.test_sgd(SGD)
    print("\n✅ All tests passed! Your SGD implementation is correct.")
