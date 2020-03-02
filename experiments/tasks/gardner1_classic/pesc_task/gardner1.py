#!/usr/bin/env python2

import math
import numpy as np

import logging


def evaluate(job_id, params):
    x = params['X']
    y = params['Y']

    f = np.cos(2.0 * x) * np.cos(y) + np.sin(x)
    c = np.cos(x) * np.cos(y) - np.sin(x) * np.sin(y) + 0.5

    # logging.info(str(dict(job_id=job_id, x=x, y=y, f=f, c=c)))

    return {
        "f": f,
        "c": -c
    }


def main(job_id, params):
    try:
        return evaluate(job_id, params)
    except Exception as ex:
        print("Exception")
        return np.nan
