#!/usr/bin/python

from __future__ import print_function

import os
import glob
import pickle
import collections
import matplotlib.pyplot as plt
import numpy as np

EPSILON = 0.01

def smooth(xs, ys, epsilon, x):
    return np.mean(ys[np.abs(xs - x) < epsilon])

def bucket_count(xs, epsilon, x):
    return np.sum(abs(xs - x) < epsilon)

def produce_traces(run):
    linear_sample_points = np.linspace(-1, 1, 500)
    return {
        "linear": linear_sample_points,
        "random_y": [
            smooth(run["random_proxy_vals"], run["random_real_vals"], EPSILON, x)
            for x in linear_sample_points
        ],
        "optimized_y": [
            smooth(run["optimized_proxy_vals"], run["optimized_real_vals"], EPSILON, x)
            for x in linear_sample_points
        ],
    }

if __name__ == "__main__":
    run_paths = glob.glob("runs/*.pickle")
    traces = collections.defaultdict(list)
    for run_path in run_paths:
        print("Processing:", run_path)
        with open(run_path, "rb") as f:
            run_data = pickle.load(f)
        for trace_name, trace_array in produce_traces(run_data).items():
            traces[trace_name].append(trace_array)

    # Average the traces.
    for k in traces:
        traces[k] = np.nanmean(traces[k], axis=0)

    plt.rcParams["figure.figsize"] = 16, 12
    plt.plot(traces["linear"], traces["random_y"])
    plt.plot(traces["linear"], traces["optimized_y"])
    plt.plot(traces["linear"], traces["optimized_y"] - traces["random_y"])
    plt.legend(["Random", "Optimized", "Difference"])
    plt.xlabel("Proxy Utility")
    plt.grid()

    plot_file = os.path.join(os.path.dirname(__file__), "output.png")
    plt.savefig(plot_file, dpi=600)
    plt.show()
