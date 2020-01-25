#!/usr/env/bin python3

import numpy as np
import torch

from .synthetic import SyntheticTestFunction


def get_feasibility_plot_2d(
    ax,
    fcn: SyntheticTestFunction,
    nx: int = 100,
    ny: int = 100,
    levels=None,
    levels_fmt='%2.1f'
):
    bounds = np.array(fcn._bounds)

    x_bounds = bounds[:, 0]
    x = np.linspace(x_bounds[0], x_bounds[1], num=nx)

    y_bounds = bounds[:, 1]
    y = np.linspace(y_bounds[0], y_bounds[1], num=ny)

    x_grid, y_grid = np.meshgrid(x, y)
    x_ = torch.tensor(x_grid.flatten())
    y_ = torch.tensor(y_grid.flatten())

    X = torch.stack((x_, y_), dim=-1)
    Z = fcn(X)

    contours = list()
    for i in range(Z.shape[1]):
        c = Z[:, i].numpy()
        c_grid = c.reshape((len(x), len(y)))

        if i > 0:
            cfill = ax.contourf(x_grid, y_grid, c_grid, levels=[0.0, np.inf], colors='lightgray')
            clines = ax.contour(x_grid, y_grid, c_grid, levels=[0.0])
            contours.append((cfill, clines))
        else:
            clines = ax.contour(x_grid, y_grid, c_grid, levels=levels)
            ax.clabel(clines, fmt=levels_fmt, colors='k')
            contours.append((clines))

    return contours
