#!/usr/env/bin python3

from typing import Optional, List, Tuple, Any, Callable
import torch
from torch import Tensor
from botorch.models.model import Model
from botorch.acquisition import AcquisitionFunction, qSimpleRegret, get_acquisition_function
from botorch.optim import optimize_acqf
from botorch.sampling import SobolQMCNormalSampler, IIDNormalSampler
from botorch.utils import get_objective_weights_transform, get_outcome_constraint_transforms

from .objective import AlboMCObjective


class AlboAcquisitionFactory(object):
    r""" ALBO acquisition function factory.

    Manages creation and fitting of an Augmented Lagrangian (AL) objective from raw objective and constraints
    using a chosen AL variant and constructs an acquisition function over the fitted AL objective.

    Provides a callable interface compatible with `get_botorch` for easy plugging with the Ax API.

    Example:
        acqf_constructor = AlboAcquisitionFactory(
            albo_objective_constructor=ClassicAlboMCObjective,      // classical squared penalty
            acquisition_function_name="qEI",                        // using EI over augmented objective
            bounds=Tensor(...),                                     // optimization bounds for inner loop
            init_mults=Tensor(...),                                 // put initial estimates for Lagrange multipliers, if available
            penalty_rate=10.0,                                      // AL penalty rate
            num_iter=10,                                            // Number of inner loop iterations
        )
    """

    _default_init_mult = 1.e-3

    def __init__(
        self,
        albo_objective_constructor: Callable[[Any], AlboMCObjective],
        acquisition_function_name: str,
        bounds: Tensor,
        init_mults: Optional[Tensor] = None,
        init_penalty_rate: float = 1.0,
        num_iter: int = 10,
        num_restarts: int = 64,
        raw_samples: int = 1024
    ) -> None:
        r"""Creates an instance.

        Args:
            albo_objective_constructor: A factory function for ALBO objective
            acquisition_function_name: A name of MC acquisition function
            bounds: [2 x d] - dim Tensor of bounds for optimization variable
            init_mults: [1 x m] - dim Tensor of initial values for Lagrangian multipliers
            init_penalty_rate: Initial penalty rate parameter for Augmented Lagrangian (default: 10.0)
            num_iter: Number of inner loop iterations for fitting Lagrangian multipliers (default: 10.0)
            num_restarts: Number of restarts for inner loop optimization
            raw_samples: Number of raw samples for initialization of inner loop optimization
        """
        self.albo_objective_constructor = albo_objective_constructor
        self.acquisition_function_name=acquisition_function_name
        self.bounds = bounds
        self.init_mults = init_mults
        self.init_penalty_rate = init_penalty_rate
        self.num_iter = num_iter
        self.num_restarts = num_restarts
        self.raw_samples = raw_samples

        self.objective_callable = None
        self.constraints_callable_list = None
        self.sampler = None

        self.albo_objective = None
        self.trace_inner = None

    def __call__(
        self,
        model: Model,
        objective_weights: Tensor,
        outcome_constraints: Tuple[Tensor, Tensor],
        X_observed: Optional[Tensor] = None,
        X_pending: Optional[Tensor] = None,
        mc_samples: int = 512,
        qmc: bool = True,
        seed: Optional[Tensor] = None,
        **kwargs: Any
    ) -> AcquisitionFunction:
        r"""Creates an acquisition function.

        The callable interface is compatible with `ax.modelbridge.factory.get_botorch` factory function.
        See for example `ax.models.torch.botorch_defaults.get_NEI`.

        Args:
            model: A fitted GPyTorch model
            objective_weights: Transform parameters from model output to raw objective
            outcome_constraints: Transform parameters from model output to raw constaints
            X_observed: Tensor of evaluated points
            X_pending: Tensor of points whose evaluation is pending
            mc_samples: The number of MC samples to use in the inner-loop optimization
            qmc: If True, use qMC instead of MC
            seed: Optional seed for MCSampler

        Returns:
            An instance of the acquisition function
        """

        self.model = model

        if qmc:
            self.sampler = SobolQMCNormalSampler(num_samples=mc_samples, seed=seed)
        else:
            self.sampler = IIDNormalSampler(num_samples=mc_samples, seed=seed)

        # Store objective and constraints
        self.objective_weights = objective_weights
        self.outcome_constraints = outcome_constraints

        # Run inner loop of Augmented Lagrangian algorithm for fitting of Lagrange Multipliers
        # and store the fitted albo objective and the trace of inner loop optimization
        # (for debugging and visualization).
        self.albo_objective, self.trace_inner = self.fit_albo_objective()

        # Create an acquisition function over the fitted AL objective
        if self.acquisition_function_name == 'qSimpleRegret':
            return qSimpleRegret(
                model=model,
                objective=self.albo_objective,
                sampler=self.sampler,
                X_pending=X_pending
            )
        else:
            return get_acquisition_function(
                acquisition_function_name=self.acquisition_function_name,
                model=model,
                objective=self.albo_objective,
                X_observed=X_observed,
                X_pending=X_pending,
                mc_samples=mc_samples,
                seed=seed,
                **kwargs
            )

    def fit_albo_objective(self) -> AlboMCObjective:
        r"""Inner loop of Augmented Lagrangian algorithm

        Args:
            model: A BoTorch model, fitted to observed data
            objective_callable: A callable transformation from model outputs to objective
            constraints_callable_list: A callable transformation from model outputs to constraints,
                with negative values imply feasibility
            sampler: An MCSampler instance for monte-carlo acquisition

        Returns:
            albo_objective: augmented objective with fitted Lagrangian multipliers
            trace: optimization trace
        """
        objective_callable = get_objective_weights_transform(self.objective_weights)
        constraints_callable_list = get_outcome_constraint_transforms(self.outcome_constraints)
        penalty_rate = self.init_penalty_rate
        num_mults = self.outcome_constraints[0].shape[0]
        if self.init_mults is not None:
            assert num_mults == self.init_mults.shape[-1]
            mults = self.init_mults
        else:
            mults = torch.Tensor([self._default_init_mult for _ in range(num_mults)])

        x_trace = torch.zeros_like(self.bounds[0].unsqueeze(0))
        mults_trace = mults.unsqueeze(0)
        output_means = torch.zeros((1, self.model.num_outputs), dtype=float)
        output_variances = torch.zeros((1, self.model.num_outputs), dtype=float)

        for i in range(self.num_iter):
            # 1. Optimize the augmented objective with fixed multipliers to find the next point for multipliers update
            albo_objective = self.albo_objective_constructor(
                objective=objective_callable,
                constraints=constraints_callable_list,
                penalty_rate=penalty_rate,
                lagrange_mults=mults
            )

            # Using predictive mean for inner loop optimization
            acq_function = qSimpleRegret(
                model=self.model,
                objective=albo_objective,
                sampler=self.sampler
            )

            x, val = optimize_acqf(
                acq_function=acq_function,
                bounds=self.bounds,
                q=1,
                num_restarts=self.num_restarts,
                raw_samples=self.raw_samples
            )

            # 2. Compute update of lagrange multipliers at the optimal point of augmented objective
            posterior = self.model.posterior(x.unsqueeze(0))
            samples = self.sampler(posterior)
            mults_next, mults_stds_next = albo_objective.get_mults_update(samples)

            # 3. Possibly apply heuristics here, i.e clamp mults before update and increase penalty rate
            mults = mults_next                          # currently not using any heuristics
            penalty_rate = self.init_penalty_rate       # seems to work just fine with a constant penalty rate

            # 4. Write trace of inner-loop optimization for debugging
            x_trace = torch.cat([x_trace, x], dim=0)
            mults_trace = torch.cat([mults_trace, mults.unsqueeze(0)], dim=0)
            output_means = torch.cat([output_means, posterior.mean.detach().squeeze(dim=0)], dim=0)
            output_variances = torch.cat([output_variances, posterior.variance.detach().squeeze(dim=0)], dim=0)

            # 5. Check stopping condition for inner loop (not implemented)
            continue

        # Construct final objective
        albo_objective = self.albo_objective_constructor(
            objective=objective_callable,
            constraints=constraints_callable_list,
            penalty_rate=penalty_rate,
            lagrange_mults=mults
        )

        trace = {
            'x': x_trace,
            'mults': mults_trace,
            'output': {
                'mean': output_means,
                'variance': output_variances
            }
        }

        return albo_objective, trace
