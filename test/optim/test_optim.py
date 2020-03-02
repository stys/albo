#!/usr/bin/env python3

import torch
from torch import Tensor

from botorch import fit_gpytorch_model
from botorch.sampling import SobolQMCNormalSampler
from botorch.utils import draw_sobol_samples

from albo.optim.optimize import AlboOptimizer, qEiAcqfOptimizer
from albo.acquisition.objective import ClassicAugmentedLagrangianMCObjective
from albo.test_functions.synthetic import GramacyTestFunction


def test_gramacy_ei():

    # noiseless Gramacy problem
    blackbox = GramacyTestFunction()

    # problem bounds
    bounds = torch.Tensor(blackbox._bounds)
    d = bounds.shape[-1]

    # augmented lagrangian objective
    objective = ClassicAugmentedLagrangianMCObjective(
        objective=lambda y: y[..., 0],
        constraints=[
            lambda y, i=i: y[..., i] for i in range(1, d)
        ],
        r=100.0
    )

    # sampler
    sampler = SobolQMCNormalSampler(
        num_samples=500,
        seed=42
    )

    # acquisition function optimizer
    acqfopt = qEiAcqfOptimizer(sampler)

    # ALBO optimizer
    opt = AlboOptimizer(
        blackbox=blackbox,
        objective=objective,
        acqfopt=acqfopt,
        sampler=sampler,
        bounds=bounds
    )

    # optimize
    opt.optimize(
        niter=40,
        init_samples=10,
        al_iter=10,
        seed=10,
        verbose=True
    )
