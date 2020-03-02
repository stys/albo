#!/usr/bin/env python3

import torch
from torch import Tensor
from albo.test_functions.synthetic import GramacyTestFunction


def test_gramacy_test_function():
    function = GramacyTestFunction()
    x = Tensor([
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [0.0, 0.0],
            [0.1951294, 0.4046587]
        ],
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [0.0, 0.0],
            [0.1951, 0.4047]
        ]
    ])
    z = function(x)
