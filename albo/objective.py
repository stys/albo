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

    An Augmented Lagrangian is a scalarized optimization objective, where `phi(t, \lambda, r)` is a non-linear
    penalty function and `lambda_i` are the estimates of Lagrange multipliers.

    ```
    L(\lambda, x, \pho) = f(x) - \sum_i \phi(c_i(x), \lambda_i, r)`
    ```

    Implementations should define a specific penalty function and its gradient with respect to constraint,
    which is used in the update rule of the Lagrange multipliers.
    """

    _default_mult = 1.e-8

    def __init__(
        self,
        objective: Callable[[Tensor], Tensor],
        constraints: List[Callable[[Tensor], Tensor]],
        penalty_rate: float = 1.0,
        lagrange_mults: Optional[Tensor] = None
    ) -> None:
        r"""A generic Augmented Lagrangian objective

        Args:
            objective: A callable mapping a `sample_shape x batch-shape x q x m`-
                dim Tensor to a `sample_shape x batch-shape x q`-dim Tensor of
                objective values.
            constraints: A list of callables, each mapping a Tensor of dimension
                `sample_shape x batch-shape x q x m` to a Tensor of dimension
                `sample_shape x batch-shape x q`, where negative values imply
                feasibility.
            penalty_rate: A rate parameter for penalty function
            lagrange_mults: A `1 x len(constraints)`-dim Tensor values of Lagrange multipliers
        """

        super(AlboMCObjective, self).__init__()
        self.objective = objective
        self.constraints = constraints
        self.penalty_rate = penalty_rate

        if lagrange_mults is not None:
            self.register_buffer("lagrange_mults", lagrange_mults.clone())
        else:
            default_lagrange_mults = torch.full((len(constraints), 1), fill_value=self._default_mult, dtype=float)
            self.register_buffer("lagrange_mults", default_lagrange_mults)

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
            penalty += self.penalty(constraint(samples), self.lagrange_mults[i], self.penalty_rate)
        return obj - penalty

    def get_mults_update(self, samples: Tensor) -> None:
        """Update rule for multipliers

        Args:
            samples: `sample_shape x batch-shape x q x m`-dim Tensor of samples
        """

        means = torch.zeros_like(self.lagrange_mults)
        stds = torch.zeros_like(self.lagrange_mults)

        for i, constraint in enumerate(self.constraints):
            z = self.grad_penalty(constraint(samples), self.lagrange_mults[i], self.penalty_rate)
            means[i] = z.mean()
            stds[i] = z.std()

        return means, stds

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
