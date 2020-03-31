#!/usr/bin/env python3

from torch import Tensor
from botorch.sampling.samplers import SobolQMCNormalSampler

from albo._acquisition.objective import ClassicAugmentedLagrangianMCObjective
from albo.optim.optimize import AlboOptimizer
from albo.test_functions.synthetic import GardnerTestFunction


def main():
    blackbox = GardnerTestFunction()

    objective = ClassicAugmentedLagrangianMCObjective(
        objective=lambda y: y[..., 0],
        constraints=[
            lambda y: y[..., 1]
        ],
        r=100.0
    )

    sampler = SobolQMCNormalSampler(num_samples=1500)

    optimizer = AlboOptimizer(
        blackbox=blackbox,
        objective=objective,
        sampler=sampler,
        bounds=Tensor(blackbox._bounds)
    )

    x_best, y_best, trace = optimizer.optimize(niter=50, init_samples=10, al_iter=10, verbose=True)

if __name__ == '__main__':
    main()

