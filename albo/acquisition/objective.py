#!/usr/bin/env python3

from abc import abstractmethod
from typing import Callable, List

import torch
from torch import Tensor

from botorch.acquisition.objective import MCAcquisitionObjective


class AugmentedLagrangianMCObjective(MCAcquisitionObjective):
    r"""Abstract class for Monte-Carlo augmented lagrangian objectives

    Assuming minimization of objective `f(x)` with constraints `c_i(x) <= 0, i = 0..m-1` a general augmented
    lagrangian objective is a sum of objective and non-linear penalties for constraints.

    `L(\lambda, x, \pho) = f(x) + \sum_i \phi(c_i(x), \lambda_i, r)`

    Given an estimate for lagrange multipliers `\lambda^k` find `x_k` as an approximate solution
    of `max_{x} L(\lambda^k, x, r)`. Next, use `x_k` to update estimates of lagrange multipliers
    according to a general equation

    `\lambda^{k+1} = \phi'(c_i(x_k), \lambda_k, r)`

    Parameter `r` is interpreted as a learning rate for proximal point method for dual problem. Variants may update
    `r` to improve convergence (citation??).

    """

    @abstractmethod
    def forward(self, samples: Tensor) -> Tensor:
        pass

    @abstractmethod
    def update_mults(self, samples: Tensor) -> None:
        pass


class ClassicAugmentedLagrangianMCObjective(AugmentedLagrangianMCObjective):
    r"""Classic augmented lagrangian with squared penalty.

    Penalty function
    \phi(c_i(x), \lambda_i, r) =
        if (\lambda_i + r c_i(x) >= 0)
        then: \lambda_i c_i(x) + (r / 2) c_i(x)^2
        else: -\lambda_i^2 / (2 r)

    Multipliers update rule
    \lambda^{k+1)_i = max(0, \lambda^{k}_i + r c_i(x_k))

    Rockafellar R.T., "The multiplier method of Hestenes and Powell applied to convex programming",
    Journal of Optimization Theory and Applications, vol. 12, 1973

    See also Nocedal J., Wright S. J., "Numerical optimization", chapter 17.4
    """

    def __init__(
        self,
        objective: Callable[[Tensor], Tensor],
        constraints: List[Callable[[Tensor], Tensor]],
        r: float = 0.1,
        mults: Tensor = None
    ) -> None:
        super(ClassicAugmentedLagrangianMCObjective, self).__init__()
        self.objective = objective
        self.constraints = constraints
        self.r = r

        if mults is None:
            mults = torch.zeros((len(constraints), 1), dtype=float)
        self.register_buffer('mults', mults)

    def forward(self, samples: Tensor) -> Tensor:
        obj = self.objective(samples)
        penalty = torch.zeros_like(obj)
        for i, constraint in enumerate(self.constraints):
            penalty += self.penalty(constraint(samples), self.mults[i], self.r)
        return - (obj + penalty)

    def update_mults(self, samples: Tensor) -> None:
        for i, constraint in enumerate(self.constraints):
            self.mults[i] = self.grad_penalty(constraint(samples), self.mults[i], self.r).mean()

    @staticmethod
    def penalty(t: Tensor, m: Tensor, r: float) -> Tensor:
        return torch.where(m + r * t < 0, - m ** 2 / (2.0 * r), m * t + (r / 2.0) * t ** 2)

    @staticmethod
    def grad_penalty(t: Tensor, m: Tensor, r: float) -> Tensor:
        return torch.max(torch.zeros_like(t, dtype=float), m + r * t)
