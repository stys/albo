#!/usr/bin/env python3

from math import pi
from typing import Optional

import torch
from torch import Tensor

from botorch.test_functions import SyntheticTestFunction, Hartmann, Branin


class GramacyTestFunction(SyntheticTestFunction):
    r""" Gramacy test function

    2-dimensional function with two constraints

        f(x) = x_1 + x_2
        c1(x) = 3./2 - x_1 - 2.*x_2 - (1./2) sin(2 pi (x_1^2 - 2 x_2))
        c2(x) = x_1^2 + x_2^2 - 3./2

    R. Gramacy et. al, Modeling an Augmented Lagrangian for Blackbox Constrained Optimization

    Note:
        Original paper provides approximate values for global optimizer and optimal value
        which is clearly not good enough, because it violates an stationarity condition of
        the lagrangian. At the optimum we should have `grad(f) = - \lambda grad(c1)`, but
        the directions of gradients are off by about 3% at their point. Below we use improved
        values. Estimated lagrange multiplier \lambda \approx 0.8595 ± 0.0003
    """

    dim = 2
    out_dim = 3
    _bounds = [[0.0, 0.0], [1.0, 1.0]]
    _optimal_value = 5.9979e-01
    _optimizers = [(0.1951294, 0.4046587)]

    def __init__(self, noise_std: Optional[float] = None) -> None:
        super(GramacyTestFunction, self).__init__(noise_std=noise_std, negate=False)

    def evaluate_true(self, X: Tensor) -> Tensor:
        x1 = X[..., 0]
        x2 = X[..., 1]
        f = x1 + x2
        c1 = (3./2) - x1 - 2.*x2 - (1./2) * torch.sin(2. * pi * (x1**2 - 2.*x2))
        c2 = x1**2 + x2**2 - 3./2
        return torch.stack([f, c1, c2], dim=-1)


class GardnerTestFunction(SyntheticTestFunction):
    r""" Garder test function

    2-dimensional function with one constraint

        f(x) = sin(x_1) + x_2
        c1(x) = sin(x_1)*sin(x_2) + 0.95

    J. Gardner et al., Bayesian Optimization with Inequality Constraints, 2014
    S. Ariafar et al., ADMMBO: Bayesian Optimization with Unknown Constraints using ADMM

    Note: there is a difference in problem specification between Gardner2014 and Ariafar2019,
    namely objective function in Gardner2014 is `f(x) = sin(x_1)` while in Ariafar2019
    it is `f(x) = sin(x_1) + x_2`
    """

    dim = 2
    out_dim = 2
    _bounds = [[0.0, 0.0], [6.0, 6.0]]
    _optimizers = [(0.0, 0.0)] # XXX

    def __init__(self, noise_std: Optional[float] = None) -> None:
        super(GardnerTestFunction, self).__init__(noise_std=noise_std, negate=False)

    def evaluate_true(self, X: Tensor) -> Tensor:
        x1 = X[..., 0]
        x2 = X[..., 1]
        f = torch.sin(x1) + x2
        c = torch.sin(x1) * torch.sin(x2) + 0.95
        return torch.stack([f, c], dim=-1)


class PichenyTestFunction(SyntheticTestFunction):
    r""" Constrained Hartman4
    """

    dim = 4
    out_dim = 2
    _bounds = [[0.0] * 4, [1.0] * 4]
    _optimizers = [(0.0, 0.0, 0.0, 0.0)] # XXX

    def __init__(self, noise_std: Optional[float] = None) -> None:
        super(PichenyTestFunction, self).__init__(noise_std=noise_std, negate=False)
        self.hartman4 = Hartmann(dim=4, noise_std=noise_std, negate=True)

    def evaluate_true(self, X: Tensor) -> Tensor:
        f = torch.sum(X, dim=-1)
        c = (self.hartman4(X))
        return torch.stack([f, c], dim=-1)


class Hartmann6Constrained(SyntheticTestFunction):
    """
    Letham2019 supplements Problem 2
    Sample problem from BoTorch tutorial https://botorch.org/tutorials/closed_loop_botorch_only
    """

    dim = 6
    out_dim = 2
    _bounds = [[0.0] * 6, [1.0] * 6]
    _optimizers = None

    def __init__(self, noise_std: Optional[float] = None) -> None:
        super(Hartmann6Constrained, self).__init__(noise_std=noise_std, negate=False)
        self.hartmann6 = Hartmann(dim=6, noise_std=noise_std, negate=False)

    def evaluate_true(self, X: Tensor) -> Tensor:
        f = self.hartmann6(X)
        c = torch.sum(X, dim=-1) - 1.0
        return torch.stack([f, c], dim=-1)


class BraninConstrained(SyntheticTestFunction):
    """
    Letham2019 supplements Problem 3
    """

    dim = 2
    out_dim = 2
    _bounds = [[-5.0, 0.0], [10.0, 15.0]]
    _optimizers = None

    def __init__(self, noise_std: Optional[float] = None) -> None:
        super(BraninConstrained, self).__init__(noise_std=noise_std, negate=False)
        self.branin = Branin(noise_std=noise_std, negate=False)

    def evaluate_true(self, X: Tensor) -> Tensor:
        f = self.branin(X) / 100.0
        c = ((X[..., 0] - 2.5)**2 + (X[..., 1] - 7.5)**2 - 50.0) / 100.0
        return torch.stack([f, c], dim=-1)


class Gardner1(SyntheticTestFunction):
    """
    Gardner2014 Simulation 1
    Letham2019 supplements Problem 4
    """

    dim = 2
    out_dim = 2
    _bounds = [[0.0, 0.0], [6.0, 6.0]]
    _optimizers = None

    def __init__(self, noise_std: Optional[float] = None) -> None:
        super(Gardner1, self).__init__(noise_std=noise_std, negate=False)

    def evaluate_true(self, X: Tensor):
        f = torch.cos(2.0 * X[..., 0]) * torch.cos(X[..., 1]) + torch.sin(X[..., 0])
        c = torch.cos(X[..., 0]) * torch.cos(X[..., 1]) - torch.sin(X[..., 0]) * torch.sin(X[..., 1]) + 0.5
        return torch.stack([f, c], dim=-1)


class Perrone(SyntheticTestFunction):
    """
    Perrone2019 toy problem
    """

    dim = 2
    out_dim = 2
    _bounds = [[-1.0, -1.0], [1.0, 1.0]]
    _optimizers = None
    x1 = -0.7
    y1 = 0.5
    x2 = 0.5
    y2 = 0.3
    x3 = -0.3
    y3 = -0.3

    def __init__(self, noise_std: Optional[float] = None) -> None:
        super(Perrone, self).__init__(noise_std=noise_std, negate=False)

    def evaluate_true(self, X: Tensor):
        f1 = (-0.7 - X[..., 0])**2 / 0.02 + 0.03
        f2 = ()