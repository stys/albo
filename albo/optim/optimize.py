#!/usr/bin/env python3

from typing import Optional, List, Tuple, Callable, TextIO

import numpy as np

import torch
from torch import Tensor

from botorch.utils import draw_sobol_samples
from botorch import fit_gpytorch_model
from botorch.optim import optimize_acqf
from botorch.models import SingleTaskGP, ModelListGP
from botorch.sampling import MCSampler
from botorch.acquisition import qSimpleRegret, qExpectedImprovement

from gpytorch.constraints import GreaterThan
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood

from ..acquisition.objective import AugmentedLagrangianMCObjective


def optimize_al_inner(
    model: ModelListGP,
    objective: AugmentedLagrangianMCObjective,
    sampler: MCSampler,
    bounds: Tensor,
    niter: int = 10,
    nprint: int = 0,
    print_file: TextIO = None
) -> (Tensor, Tensor):
    """ Inner loop Augmented Lagrangian iterations
    """

    acqfn = qSimpleRegret(
        model=model,
        sampler=sampler,
        objective=objective
    )

    trace = list()

    for i in range(niter):
        x, L = optimize_acqf(
            acq_function=acqfn,
            bounds=bounds,
            q=1,
            num_restarts=1,
            raw_samples=500
        )

        samples = sampler(model.posterior(x))
        objective.update_mults(samples)

        x_ = x.detach().numpy()[0]
        L_ = L.detach().numpy()
        mean_ = model.posterior(x).mean.detach().numpy()[0]
        mults_ = objective.mults.T.detach().numpy()[0]

        trace.append(dict(
            x=x_,
            mults=mults_,
            L=L_
        ))

        if nprint > 0 and i % nprint == 0:
            print(
                'Iter inner:', i,
                'x:', x.numpy(),
                'y (mean):', mean_,
                'al:', L_,
                'mults:', mults_,
                file=print_file
            )

    return x, L, trace


def optimize_al_ei(
    model: ModelListGP,
    objective: AugmentedLagrangianMCObjective,
    sampler: MCSampler,
    best_f: float,
    bounds: Tensor,
    num_restarts: int = 10,
    raw_samples: int = 500,
    verbose: bool = False
) -> Tensor:
    """ Outer loop step: optimization of EI over Augmented Lagrangian objective
    """

    ei = qExpectedImprovement(
        model=model,
        best_f=best_f,
        sampler=sampler,
        objective=objective
    )

    x, f = optimize_acqf(
        acq_function=ei,
        bounds=bounds,
        q=1,
        num_restarts=num_restarts,
        raw_samples=raw_samples
    )

    return x, f


class AlboOptimizer(object):
    def __init__(self,
        blackbox: Callable[[Tensor], Tensor],
        objective: AugmentedLagrangianMCObjective,
        sampler: MCSampler,
        bounds: Tensor,
        min_noise: float = 1.e-5
    ) -> None:
        self.blackbox = blackbox
        self.objective = objective
        self.sampler = sampler
        self.bounds = bounds
        self.min_noise = min_noise
        self.model = None
        self.trace = None

    def generate_initial_data(self, nsamples=10, seed=None):
        x_train = draw_sobol_samples(self.bounds, n=1, q=nsamples, seed=seed)[0]
        y_train = self.blackbox(x_train)
        return x_train, y_train

    def initialize_model(self, x_train, y_train, state_dict=None):
        n = y_train.shape[-1]
        gp_models = []
        for i in range(n):
            y = y_train[..., i].unsqueeze(-1)
            gp_model = SingleTaskGP(train_X=x_train, train_Y=y)
            gp_model.likelihood.noise_covar.register_constraint("raw_noise", GreaterThan(self.min_noise))
            gp_models.append(gp_model)
        model_list = ModelListGP(*gp_models)
        mll = SumMarginalLogLikelihood(model_list.likelihood, model_list)
        if state_dict is not None:
            model_list.load_state_dict(state_dict)
        return mll, model_list

    def optimize(
        self,
        niter: int,
        init_samples: int = 10,
        al_iter: int = 10,
        seed: Optional[int] = None,
        verbose: bool = False,
        print_file: TextIO = None
    ):
        """ Closed loop optimization
        """
        x_al = torch.zeros((init_samples + niter, self.bounds.shape[0]))
        mults = torch.zeros((init_samples + niter, self.objective.mults.shape[0]))
        traces_inner = list()

        idx_best = np.zeros((init_samples + niter), dtype=int)
        i_best = -1
        f_best = np.inf

        # initial exploration
        x, y = self.generate_initial_data(nsamples=init_samples, seed=seed)

        # update trace
        for i, y_next in enumerate(y):
            if torch.prod(y_next[1:] < 0.0) and y_next[0] < f_best:
                i_best = i
                f_best = y_next[0]
            idx_best[i] = i_best
            traces_inner.append(None)

        # outer loop
        for i_ in range(niter):
            i = i_ + init_samples

            # fit GPs
            mll, self.model = self.initialize_model(x, y)
            fit_gpytorch_model(mll)

            # inner loop
            self.objective.reset_mults()   # this is very important

            x_inner, al, trace_inner = optimize_al_inner(
                model=self.model,
                objective=self.objective,
                sampler=self.sampler,
                bounds=self.bounds,
                niter=al_iter,
                print_file=print_file,
                nprint=1 if verbose else 0
            )

            # optimize EI
            x_next, ei = optimize_al_ei(
                model=self.model,
                objective=self.objective,
                sampler=self.sampler,
                best_f=al,
                bounds=self.bounds
            )
            y_next = self.blackbox(x_next)

            x = torch.cat([x, x_next], dim=0)
            y = torch.cat([y, y_next], dim=0)

            # update trace
            if torch.prod(y_next[0, 1:] < 0.0) and y_next[0, 0] < f_best:
                i_best = i
                f_best = y_next[0, 0]
            idx_best[i] = i_best
            x_al[i, :] = x_inner
            mults[i, :] = self.objective.mults.T
            traces_inner.append(trace_inner)

            if verbose:
                print(
                    'Iter outer:', i,
                    'x:', x_next.numpy(),
                    'y:', y_next.numpy(),
                    'mults:', mults[i].detach().numpy(),
                    'best iter:', i_best,
                    'best x:', x[i_best].numpy(),
                    'best y:', y[i_best].numpy(),
                    file=print_file
                )

            self.trace = dict(
                x=x,
                y=y,
                idx_best=idx_best,
                x_al=x_al,
                mults=mults,
                traces_inner=traces_inner
            )

        if i_best >= 0:
            return x[i_best], y[i_best], self.trace
        else:
            return None, None, self.trace
