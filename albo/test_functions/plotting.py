#!/usr/env/bin python3

import numpy as np
import torch

from .synthetic import SyntheticTestFunction


def get_feasibility_plot_2d(ax, fcn: SyntheticTestFunction, nx: int = 100, ny: int = 100):
    bounds = np.array(fcn._bounds)

    x_bounds = bounds[:, 0]
    x = np.linspace(x_bounds[0], x_bounds[1], num=nx)

    y_bounds = bounds[:, 0]
    y = np.linspace(y_bounds[0], y_bounds[1], num=ny)

    x_grid, y_grid = np.meshgrid(x, y)
    x_ = torch.tensor(x_grid.flatten())
    y_ = torch.tensor(y_grid.flatten())

    X = torch.stack((x_, y_), dim=-1)
    Z = fcn(X)

    contours = list()
    for i in range(1, Z.shape[1]):
        c = Z[:, i].numpy()
        c_grid = c.reshape((len(x), len(y)))
        contours.append(ax.contour(x_grid, y_grid, c_grid, levels=[0.0]))

    return contours
