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

def produce_trace(run):
    linear = np.linspace(-1, 1, NUM_BUCKETS)

    trace = collections.defaultdict(list)
    trace["linear"] = linear

    for x in linear:
        random_proxy_bucket = make_bucket(run["random_proxy_vals"], EPSILON, x)
        trace["random_samples"].append(bucket_count(random_proxy_bucket))
        trace["random_real_vs_proxy"].append(bucket_mean(run["random_real_vals"], random_proxy_bucket))
        trace["random_proxy"].append(bucket_mean(run["random_proxy_vals"], random_proxy_bucket))

        optimized_bucket = make_bucket(run["optimized_proxy_vals"], EPSILON, x)
        trace["optimized_samples"].append(bucket_count(optimized_bucket))
        trace["optimized_real"].append(bucket_mean(run["optimized_real_vals"], optimized_bucket))
        trace["optimized_proxy"].append(bucket_mean(run["optimized_proxy_vals"], optimized_bucket))

        delta_bucket = make_bucket(run["optimized_proxy_vals"] - run["random_proxy_vals"], EPSILON, x)
        trace["deltas"].append(bucket_mean(run["optimized_real_vals"] - run["random_real_vals"], delta_bucket))
        trace["delta_samples"].append(bucket_count(delta_bucket))

        if run["hyper_parameters"]["INPUT_SIZE"] == 1:
            samples_bucket = make_bucket(run["samples"], EPSILON, x)
            trace["real_values"].append(bucket_mean(run["random_real_vals"], samples_bucket))
            trace["proxy_values"].append(bucket_mean(run["random_proxy_vals"], samples_bucket))
            trace["optimized_count"].append(bucket_count(make_bucket(run["optimized_samples"], EPSILON, x)))

        # randoptimized_bucket = make_bucket(run["randoptimized_proxy_vals"], EPSILON, x)
        # trace["randoptimized_proxy"].append(bucket_mean(run["randoptimized_proxy_vals"], randoptimized_bucket))
        # trace["randoptimized_real"].append(bucket_mean(run["randoptimized_real_vals"], randoptimized_bucket))
        # randoptimized_samples.append(bucket_count(randoptimized_bucket))

        # random_sampled_proxy_bucket = make_bucket(run["random_sampled_proxy_vals"], EPSILON, x)
        # trace["random_real_vs_sampled_proxy"].append(bucket_mean(run["random_real_vals"], random_sampled_proxy_bucket))
        # trace["random_sampled_proxy"].append(bucket_mean(run["random_sampled_proxy_vals"], random_sampled_proxy_bucket))

        # sampled_proxy_optimized_bucket = make_bucket(run["sampled_proxy_optimized_proxy_vals"], EPSILON, x)
        # trace["sampled_proxy_optimized_samples"].append(bucket_count(sampled_proxy_optimized_bucket))
        # trace["sampled_proxy_optimized_real"].append(bucket_mean(run["sampled_proxy_optimized_real_vals"], sampled_proxy_optimized_bucket))
        # trace["sampled_proxy_optimized_proxy"].append(bucket_mean(run["sampled_proxy_optimized_proxy_vals"], sampled_proxy_optimized_bucket))

        random_grad_opt_proxy_bucket = make_bucket(run["random_grad_opt_proxy_vals"], EPSILON, x)
        trace["random_real_vs_grad_opt_proxy"].append(bucket_mean(run["random_real_vals"], random_grad_opt_proxy_bucket))
        trace["random_grad_opt_proxy"].append(bucket_mean(run["random_grad_opt_proxy_vals"], random_grad_opt_proxy_bucket))

        grad_opt_proxy_optimized_bucket = make_bucket(run["grad_opt_proxy_optimized_proxy_vals"], EPSILON, x)
        trace["grad_opt_proxy_optimized_samples"].append(bucket_count(grad_opt_proxy_optimized_bucket))
        trace["grad_opt_proxy_optimized_real"].append(bucket_mean(run["grad_opt_proxy_optimized_real_vals"], grad_opt_proxy_optimized_bucket))
        trace["grad_opt_proxy_optimized_proxy"].append(bucket_mean(run["grad_opt_proxy_optimized_proxy_vals"], grad_opt_proxy_optimized_bucket))

    return trace

def get_trace_for_file(run_path):
    print("Processing:", run_path)
    with open(run_path, "rb") as f:
        return produce_trace(pickle.load(f))


if __name__ == "__main__":
    run_paths = glob.glob("runs/*.pickle")
    traces = collections.defaultdict(list)
    for run_path in run_paths:
        trace = get_trace_for_file(run_path)
        for trace_name, trace_list in trace.items():
            trace_array = np.asarray(trace_list)
            traces[trace_name].append(trace_array)

    # Average the traces.
    for k in traces:
        traces_k_array = np.asarray(traces[k])
        traces[k] = np.nanmean(traces_k_array, axis=0)

    plt.figure(figsize=(16, 12))

    plt.plot(traces["linear"], traces["random_real_vs_proxy"], label="Real Utility vs. Proxy Utility (Random Data)")
    plt.plot(traces["linear"], traces["optimized_real"], label="Real Utility vs. Proxy Utility (Optimized Data)")

    plt.plot(traces["linear"], traces["optimized_real"] - traces["random_real_vs_proxy"], label="Real Utility vs. Proxy Utility (Optimized Data - Random Data) [Goodhart Error]")

    # plt.plot(traces["linear"], traces["optimized_proxy"] - traces["random_proxy"], label="Proxy Utility (Optimized - Random) [Bucket Error]")

    # plt.plot(traces["linear"], traces["deltas"], label="Opt Real - Rand Real vs. Opt Proxy - Rand Proxy [Optimization Delta]")

    # plt.plot(traces["linear"], norm(traces["random_samples"]), label="Random Samples")
    # plt.plot(traces["linear"], norm(traces["optimized_samples"]), label="Optimized Samples")
    # plt.plot(traces["linear"], norm(traces["randoptimized_samples"]), label="Randoptimized Samples")
    # plt.plot(traces["linear"], traces["randoptimized_real"], label="Real Utility (Randoptimized)")
    # plt.plot(traces["linear"], norm(traces["delta_samples"]), label="Delta Samples")

    # plt.plot(traces["linear"], traces["real_values"], label="Real Utility")
    # plt.plot(traces["linear"], traces["proxy_values"], label="Proxy Utility")
    # plt.plot(traces["linear"], norm(traces["optimized_count"]), label="Optimized Points")

    # plt.plot(traces["linear"], traces["random_real_vs_sampled_proxy"], label="Real Utility vs. Sampled Proxy Utility (Random Data)")
    # plt.plot(traces["linear"], traces["sampled_proxy_optimized_real"], label="Real Utility vs. Sampled Proxy Utility (Sampled Proxy Optimized Data)")
    # plt.plot(traces["linear"], traces["sampled_proxy_optimized_real"] - traces["random_real_vs_sampled_proxy"], label="Real Utility vs. Sampled Proxy Utility (Sampled Proxy Optimized Data - Random Data) [Goodhart Error]")

    plt.plot(traces["linear"], traces["random_real_vs_grad_opt_proxy"], label="Real Utility vs. Grad Opt Proxy Utility (Random Data)")
    plt.plot(traces["linear"], traces["grad_opt_proxy_optimized_real"], label="Real Utility vs. Grad Opt Proxy Utility (Grad Opt Proxy Optimized Data)")
    plt.plot(traces["linear"], traces["grad_opt_proxy_optimized_real"] - traces["random_real_vs_grad_opt_proxy"], label="Real Utility vs. Grad Opt Proxy Utility (Grad Opt Proxy Optimized Data - Random Data) [Goodhart Error]")

    plt.legend()
    plt.xlabel("Proxy Utility")
    # plt.xlabel("Input Value")
    plt.grid()

    plot_file = os.path.join(os.path.dirname(__file__), "output.png")
    plt.savefig(plot_file)
    plt.show()
