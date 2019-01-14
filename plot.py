#!/usr/bin/python

from __future__ import print_function

import sys
import os
import glob
import pickle
import collections
import matplotlib.pyplot as plt
import numpy as np

if sys.version_info < (3,):
    range = xrange


EPSILON = 0.01
NUM_BUCKETS = 500


def make_bucket(bucket_vals, epsilon, x):
    return np.abs(bucket_vals - x) < epsilon

def bucket_mean(bucket_vals, bucket):
    return np.mean(bucket_vals[bucket])

def sample_delta(samples, optimized_samples, bucket):
    return np.mean(optimized_samples[bucket] - samples[bucket])

def bucket_count(bucket):
    count = np.sum(bucket)
    return count if count != 0 else np.nan

# def expand_bucket(bucket):
#     first_true_i = np.argmax(bucket)
#     if not first_true_i:
#         return bucket
#     new_bucket = np.copy(bucket)
#     new_bucket[first_true_i - 1] = True
#     return new_bucket

# def sign(x):
#     return x / abs(x)

def norm(arr):
    return arr/np.nanmax(arr)

def get_traces_for_file(run_path):
    print("Processing:", run_path)
    with open(run_path, "rb") as f:
        return pickle.load(f)

def produce_traces(run):
    linear = np.linspace(-1, 1, NUM_BUCKETS)

    random_proxy = []
    optimized_proxy = []
    # randoptimized_proxy = []

    random_real = []
    optimized_real = []
    # randoptimized_real = []

    random_samples = []
    optimized_samples = []
    # randoptimized_samples = []

    deltas = []
    delta_samples = []

    real_values = []
    proxy_values = []
    optimized_count = []

    for x in linear:
        random_bucket = make_bucket(run["random_proxy_vals"], EPSILON, x)
        optimized_bucket = make_bucket(run["optimized_proxy_vals"], EPSILON, x)
        # randoptimized_bucket = make_bucket(run["randoptimized_proxy_vals"], EPSILON, x)

        random_proxy.append(bucket_mean(run["random_proxy_vals"], random_bucket))
        optimized_proxy.append(bucket_mean(run["optimized_proxy_vals"], optimized_bucket))
        # randoptimized_proxy.append(bucket_mean(run["randoptimized_proxy_vals"], randoptimized_bucket))

        random_real.append(bucket_mean(run["random_real_vals"], random_bucket))
        optimized_real.append(bucket_mean(run["optimized_real_vals"], optimized_bucket))
        # randoptimized_real.append(bucket_mean(run["randoptimized_real_vals"], randoptimized_bucket))

        optimized_samples.append(bucket_count(optimized_bucket))
        random_samples.append(bucket_count(random_bucket))
        # randoptimized_samples.append(bucket_count(randoptimized_bucket))

        delta_bucket = make_bucket(run["optimized_proxy_vals"] - run["random_proxy_vals"], EPSILON, x)
        deltas.append(bucket_mean(run["optimized_real_vals"] - run["random_real_vals"], delta_bucket))
        delta_samples.append(bucket_count(delta_bucket))

        samples_bucket = make_bucket(run["samples"], EPSILON, x)
        real_values.append(bucket_mean(run["random_real_vals"], samples_bucket))
        proxy_values.append(bucket_mean(run["random_proxy_vals"], samples_bucket))
        optimized_count.append(bucket_count(make_bucket(run["optimized_samples"], EPSILON, x)))

    return {
        var: val for var, val in locals().items()
        if var in (
            "linear",
            "random_proxy",
            "optimized_proxy",
            "random_real",
            "optimized_real",
            "random_samples",
            "optimized_samples",
            "deltas",
            "delta_samples",
            "real_values",
            "proxy_values",
            "optimized_count",
            # "randoptimized_samples",
            # "randoptimized_proxy",
            # "randoptimized_real",
        )
    }


if __name__ == "__main__":
    run_paths = glob.glob("runs/*.pickle")
    traces = collections.defaultdict(list)
    for run_path in run_paths:
        run_data = get_traces_for_file(run_path)
        for trace_name, trace_array in produce_traces(run_data).items():
            traces[trace_name].append(trace_array)

    # Average the traces.
    for k in traces:
        traces[k] = np.nanmean(traces[k], axis=0)

    plt.figure(figsize=(16, 12))

    plt.plot(traces["linear"], traces["random_real"], label="Real Utility (Random)")
    plt.plot(traces["linear"], traces["optimized_real"], label="Real Utility (Optimized)")

    plt.plot(traces["linear"], traces["optimized_real"] - traces["random_real"], label="Real Utility (Optimized - Random)")

    plt.plot(traces["linear"], traces["optimized_proxy"] - traces["random_proxy"], label="Proxy Utility (Optimized - Random)")

    plt.plot(traces["linear"], traces["deltas"], label="Optimized - Random (Real vs. Proxy)")

    # plt.plot(traces["linear"], norm(traces["random_samples"]), label="Random Samples")
    # plt.plot(traces["linear"], norm(traces["optimized_samples"]), label="Optimized Samples")
    # plt.plot(traces["linear"], norm(traces["randoptimized_samples"]), label="Randoptimized Samples")
    # plt.plot(traces["linear"], traces["randoptimized_real"], label="Real Utility (Randoptimized)")
    # plt.plot(traces["linear"], norm(traces["delta_samples"]), label="Delta Samples")

    # plt.plot(traces["linear"], traces["real_values"], label="Real Utility")
    # plt.plot(traces["linear"], traces["proxy_values"], label="Proxy Utility")
    # plt.plot(traces["linear"], norm(traces["optimized_count"]), label="Optimized Points")

    plt.legend()
    plt.xlabel("Proxy Utility")
    # plt.xlabel("Input Value")
    plt.grid()

    plot_file = os.path.join(os.path.dirname(__file__), "output.png")
    plt.savefig(plot_file)
    plt.show()
