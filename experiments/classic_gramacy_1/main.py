#!/usr/bin/python3

import numpy as np

import torch
from torch import Tensor
from torch.quasirandom import SobolEngine

from botorch import fit_gpytorch_model
from botorch.optim import optimize_acqf
from botorch.acquisition import qSimpleRegret, LinearMCObjective, qExpectedImprovement
from botorch.models import SingleTaskGP, ModelListGP
from botorch.sampling.samplers import SobolQMCNormalSampler

from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
from gpytorch.constraints import GreaterThan

from albo.test_functions.synthetic import GramacyTestFunction
from albo.acquisition.objective import AugmentedLagrangianMCObjective, ClassicAugmentedLagrangianMCObjective


def generate_initial_data(nsamples=10, noise_std=None, seed=None):
    sobol_engine = SobolEngine(dimension=2, scramble=True, seed=seed)
    x = sobol_engine.draw(n=nsamples,dtype=torch.float)
    function = GramacyTestFunction(noise_std=noise_std)
    z = function(x)
    return x, z


def initialize_model(x, z, state_dict=None):
    n = z.shape[-1]
    gp_models = []
    for i in range(n):
        y = z[..., i].unsqueeze(-1)
        gp_model = SingleTaskGP(train_X=x, train_Y=y)
        gp_model.likelihood.noise_covar.register_constraint("raw_noise", GreaterThan(1e-5))
        gp_models.append(gp_model)
    model_list = ModelListGP(*gp_models)
    mll = SumMarginalLogLikelihood(model_list.likelihood, model_list)
    if state_dict is not None:
        model_list.load_state_dict(state_dict)
    return mll, model_list


def fit_augmented_objective(model, augmented_objective: AugmentedLagrangianMCObjective, x_train: Tensor, y_train: Tensor):
    sampler = SobolQMCNormalSampler(num_samples=1500)

    print()
    print("Optimizing augmented lagrangian on GP surrogate")
    print(f"x_1\tx_2\tf_est\tc1_est\tc2_est\tmult1\tmult2")
    for i in range(5):
        acqfn = qSimpleRegret(
            model=model,
            sampler=sampler,
            objective=augmented_objective
        )

        canditate, _ = optimize_acqf(
            acq_function=acqfn,
            bounds=Tensor([[0.0, 0.0], [1.0, 1.0]]),
            q=1,
            num_restarts=1,
            raw_samples=500
        )

        x = canditate.detach()
        samples = sampler(model.posterior(x))
        augmented_objective.update_mults(samples)
        augmented_objective.r = 100.0

        x_ = x.numpy()[0]
        acqfn_ = acqfn(x).detach().numpy()[0]
        pred_ = model.posterior(x).mean.detach().numpy()[0]
        mults_ = augmented_objective.mults.detach().numpy()
        print(f"{x_[0]:>6.4f}\t{x_[1]:>6.4f}\t{pred_[0]:>6.4f}\t{pred_[1]:>6.4f}\t{pred_[2]:>6.4f}\t{mults_[0][0]:>6.4f}\t{mults_[1][0]:>6.4f}")

    f_best = augmented_objective(model.posterior(x).mean)

    ei = qExpectedImprovement(
        model=model,
        best_f=f_best,
        sampler=sampler,
        objective=augmented_objective #linearized_objective
    )

    for i in range(x_train.shape[0]):
        xx = x_train[i, :]
        print(i, xx, ei(xx.unsqueeze(-1).T), f_best, augmented_objective(y_train[i, :].unsqueeze(-1).T), y_train[i, :])

    canditate, _ = optimize_acqf(
        acq_function=ei,
        bounds=Tensor([[0.0, 0.0], [1.0, 1.0]]),
        q=1,
        num_restarts=10,
        raw_samples=500
    )

    x_new = canditate.detach()

    x_new_ = canditate.detach().numpy()[0]
    f_best_ = f_best.detach().numpy()[0]
    f_new_ = augmented_objective(model.posterior(x_new).mean).detach().numpy()[0]
    ei_new_ = ei(x_new).detach().numpy()[0]
    print()
    print("Optimizing EI on linearized objective")
    print(f"x_1\tx_2\tf_best\tf_new_\tei")
    print(
        f"{x_new_[0]:>6.4f}\t",
        f"{x_new_[1]:>6.4f}\t",
        f"{f_best_:>6.4f}\t",
        f"{f_new_:>6.4f}\t",
        f"{ei_new_:>6.4f}"
    )

    return x_new


def main():
    function = GramacyTestFunction()

    for j in range(25):

        x, y = generate_initial_data()
        for i in range(50):
            mll, model = initialize_model(x, y)
            fit_gpytorch_model(mll)

            augmented_objective = ClassicAugmentedLagrangianMCObjective(
                objective=lambda y: y[..., 0],
                constraints=[
                    lambda y: y[..., 1],
                    lambda y: y[..., 2]
                ]
            )

            x_new = fit_augmented_objective(model, augmented_objective, x, y)
            y_new = function(x_new)

            x = torch.cat([x, x_new], dim=0)
            y = torch.cat([y, y_new], dim=0)

        np.save(f'results/x_{j}.bin', x)
        np.save(f'results/y_{j}.bin', y)

if __name__ == '__main__':
    main()
