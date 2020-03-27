#!/usr/bin/env python2

import math
import numpy as np


def evaluate(job_id, params):
    x = params['X']
    y = params['Y']

    # print('Evaluating at (%f, %f)' % (x, y))

    f = float(np.square(y - (5.1 / (4 * np.square(math.pi))) * np.square(x) + (5 / math.pi) * x - 6) + 10 * (1 - (1. / (8 * math.pi))) * np.cos(x) + 10)
    c = (x - 2.5) ** 2 + (y - 7.5) ** 2 - 50.0

    return {
        "f": f / 100.0,
        "c": - c / 100.0
    }


def main(job_id, params):
    try:
        return evaluate(job_id, params)
    except Exception as ex:
        print("Exception")
        return np.nan
