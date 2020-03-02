#!/usr/bin/env python3

from functools import partial

from torch import Tensor
from botorch.sampling import SobolQMCNormalSampler

from albo.test_functions.synthetic import Gardner1
from albo.acquisition.objective import ClassicAugmentedLagrangianMCObjective
from albo.optim.optimize import AlboOptimizer, qEiAcqfOptimizer


class Task():
    pass


class TaskGenerator(object):
    def __init__(self, inject_modules: object = None, conf: dict = None):
        self.inject_modules = inject_modules
        self.conf = conf

    def get_task_runner(self, param, **kw):
        blackbox = Gardner1(noise_std=param.noise_std)
        bounds = Tensor(blackbox._bounds)

        objective = ClassicAugmentedLagrangianMCObjective(
            objective=lambda y: y[..., 0],
            constraints=list(lambda y, i=j: y[..., i] for j in range(1, blackbox.out_dim)),
            r=param['r']
        )

        sampler = SobolQMCNormalSampler(
            num_samples=param.num_mc_samples,
            seed=param.get('seed', None)
        )

        acqfopt = qEiAcqfOptimizer(sampler=sampler)

        optimizer = AlboOptimizer(
            blackbox=blackbox,
            objective=objective,
            acqfopt=acqfopt,
            sampler=sampler,
            bounds=bounds
        )

        run = partial(
            optimizer.optimize,
            niter=param.num_iter,
            init_samples=param.num_init_samples,
            al_iter=param.num_al_iter,
            seed=param.get('seed', None)
        )

        return run

    def generate(self):
        for key, param in self.conf.items():
            task = Task()
            task.key = key
            task.param = param
            task.run = lambda **kw: self.get_task_runner(param)(**kw)

            yield task
