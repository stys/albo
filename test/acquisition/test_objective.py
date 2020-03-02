#!/usr/bin/env python3

import torch
from torch import Tensor

from albo.acquisition.objective import ClassicAugmentedLagrangianMCObjective


def test_classic_augmented_lagrangian_mc_objective():
    objective = lambda z: z[..., 0]
    constraint1 = lambda z: z[..., 1]
    constraint2 = lambda z: z[..., 2]
    constraints = [constraint1, constraint2]
    mults = torch.ones((len(constraints), 1), dtype=float)

    classic_augmented_lagrangian_mc_objective = ClassicAugmentedLagrangianMCObjective(
        objective=objective,
        constraints=constraints,
        r=100.0,
        mults=mults
    )

    samples = Tensor([
        [[1.0, -0.0, -0.1]],
        [[2.0, 0.1, 0.5]],
        [[3.0, -0.5, 0.5]],
        [[5.0, -1.0, -1.0]],
        [[10.0, -2.5, -5.0]]
    ])

    values = classic_augmented_lagrangian_mc_objective(samples)
    classic_augmented_lagrangian_mc_objective.update_mults(samples)

    assert True
