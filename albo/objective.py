#!/usr/bin/env python3

from abc import abstractmethod
from typing import List, Callable, Optional

import torch
from torch import Tensor
from botorch.acquisition.objective import MCAcquisitionObjective


class AlboMCObjective(MCAcquisitionObjective):
    r"""Base class for Monte-Carlo augmented lagrangian objectives

    A maximization problem with constraints
    ```
    f(x) -> max., s.t.
    c_i(x) <= 0, i = 0..m-1`
    x \in B
    ```

    An Augmented Lagrangian is a scalarized optimization objective, where `phi(t)` are non-linear
    penalty functions and `lambda_i` are estimates of Lagrange multipliers.
    ```
    L(\lambda, x, \pho) = f(x) - \sum_i \phi(c_i(x), \lambda_i, r)`
    ```

    Implementations must define a penalty function and an update rule for multipliers.
    """

    _default_inital_mult = 1.e-8

    def __init__(
        self,
        objective: Callable[[Tensor], Tensor],
        constraints: List[Callable[[Tensor], Tensor]],
        rate: float = 1.0,
        init_mults: Optional[Tensor] = None
    ) -> None:
        r"""Feasibility-weighted objective.

        Args:
            objective: A callable mapping a `sample_shape x batch-shape x q x m`-
                dim Tensor to a `sample_shape x batch-shape x q`-dim Tensor of
                objective values.
            constraints: A list of callables, each mapping a Tensor of dimension
                `sample_shape x batch-shape x q x m` to a Tensor of dimension
                `sample_shape x batch-shape x q`, where negative values imply
                feasibility.
            rate: A rate parameter for penalty function
            init_mults: A `1 x len(constraints)`-dim Tensor with initial values
                of Lagrange multipliers
        """

        super(AlboMCObjective, self).__init__()
        self.objective = objective
        self.constraints = constraints
        self.rate = rate

        if init_mults is not None:
            self.init_mults = init_mults
        else:
            self.init_mults = torch.full((len(constraints), 1), fill_value=self._default_inital_mult, dtype=float)

        self.register_buffer('mults', self.init_mults.clone())

    def forward(self, samples: Tensor) -> Tensor:
        r"""Evaluate augmented objective on the samples

        Args:
            samples: A `sample_shape x batch_shape x q x m`-dim Tensors of
                samples from a model posterior.

        Returns:
            A `sample_shape x batch_shape x q`-dim Tensor of augmented
            objective values (assuming maximization).

        """
        obj = self.objective(samples)
        penalty = torch.zeros_like(obj)
        for i, constraint in enumerate(self.constraints):
            penalty += self.penalty(constraint(samples), self.mults[i], self.rate)
        return obj - penalty

    def reset_mults(self, mults: Optional[Tensor] = None) -> None:
        r"""Reset Lagrange multipliers to default values or to a given value

        Args:
            mults: A `1 x len(constraints)`-dim Tensor with values
                of Lagrange multipliers

        """
        if mults is not None:
            self.mults = mults
        else:
            self.mults = self.init_mults.clone()

    def update_mults(self, samples: Tensor) -> None:
        r"""
        """

        def update_mults(self, samples: Tensor) -> None:
            for i, constraint in enumerate(self.constraints):
                self.mults[i] = self.grad_penalty(constraint(samples), self.mults[i], self.r).mean()

    @abstractmethod
    def penalty(self, t: Tensor, m: float, r: float) -> Tensor:
        r"""Abstract penalty function

        Args:
            t: `sample_shape x batch_shape x q`-dim Tensor of constraint samples
            m: value of Lagrange multiplier
            r: penalty rate parameter

        Returns:
            A `sample_shape x batch_shape x q`-dim Tensor of penalties.

        """
        pass

    @abstractmethod
    def grad_penalty(self, t: Tensor, m: float, r: float):
        r"""Gradient of penalty function

        Args:
            t: `sample_shape x batch_shape x q`-dim Tensor of constraint samples
            m: value of Lagrange multiplier
            r: penalty rate parameter

        Returns:
            A `sample_shape x batch_shape x q`-dim Tensor of penalties gradients
        """
        pass


class ClassicAlboMCObjective(AlboMCObjective):
    r"""Augmented Lagrangian objective with classical square penalty fuction

    [Rockafellar1973]
    """

    def penalty(self, t: Tensor, m: float, r: float):
        return torch.where(m + r * t < 0, - m ** 2 / (2.0 * r), m * t + (r / 2.0) * t ** 2)

    def grad_penalty(self, t: Tensor, m: Tensor, r: float) -> Tensor:
        return torch.max(torch.zeros_like(t, dtype=float), m + r * t)


class ExpAlboMCObjective(AlboMCObjective):
    r"""Augmented Lagrangian objective with exponential penalty function

    [Bertsekas1982, Iusem1999]
    """

    def penalty(self, t: Tensor, m: Tensor, r: float) -> Tensor:
        return (m / r) * (torch.exp(r * t) - 1.0)

    def grad_penalty(self, t: Tensor, m: Tensor, r: float) -> Tensor:
        return m * torch.exp(r * t)


class SmoothedAlboMCObjective(AlboMCObjective):
    r"""Augmented Lagrangian objective based on smooth penalty variant

    This penalty function is given by M. Zibulevsky in his lectures on Augmented Lagrangian method
    https://www.youtube.com/watch?v=POPYQLG6n00
    https://www.youtube.com/watch?v=3MrlbUoO1y4
    """

    def penalty(self, t: Tensor, m: Tensor, r: float) -> Tensor:
        x = r * t
        y = x * x * 0.5 + x
        z = - torch.log(-2.0 * torch.min(torch.full_like(x, -0.5), x)) / 4.0 - 3. / 8.
        return m * torch.where(x > -0.5, y, z) / r

    def grad_penalty(self, t: Tensor, m: Tensor, r: float) -> Tensor:
        x = r * t
        y = x + 1
        z = - 0.25 / x
        return m * torch.where(x > -0.5, y, z)
