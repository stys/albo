#!/usr/bin/env python3

from typing import List, Dict, Tuple, Optional, Any

import numpy as np
import pandas as pd

from ax.core import Arm, Data, Experiment, GeneratorRun, Observation, ObservationData, ObservationFeatures
from ax.modelbridge import ModelBridge


def get_untransformed_trace(
    experiment: Experiment,
    model: ModelBridge,
    trace: Any
):
    observation_features = []
    observation_data = []

    metric_names = list(m for m in model.model.metric_names if m in experiment.optimization_config.metrics)

    for i, x in enumerate(trace['x']):
        observation_features.append(
            ObservationFeatures(
                parameters={
                    p: x[i].item() for i, p in enumerate(model.parameters)
                }
            )
        )

        output_means = trace['output']['mean'][i, :]
        output_variances = trace['output']['variance'][i, :]
        observation_data.append(
            ObservationData(
                metric_names=metric_names,
                means=output_means.numpy(),
                covariance=np.diag(output_variances.numpy())
            )
        )

    for transform_name, transform in reversed(model.transforms.items()):
        observation_features = transform.untransform_observation_features(observation_features)
        observation_data = transform.untransform_observation_data(observation_data, observation_features)

    return list(Observation(f, d) for f, d in zip(observation_features, observation_data))


def df_from_observations(
    observations: List[Observation]
):
    pd.DataFrame(data=observations)
