#!/usr/bin/python

import glob, pickle, collections
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
			smooth(run["random_xs"], run["random_ys"], EPSILON, x)
			for x in linear_sample_points
		],
		"optimized_y": [
			smooth(run["optimized_xs"], run["optimized_ys"], EPSILON, x)
			for x in linear_sample_points
		],
	}

if __name__ == "__main__":
	run_paths = glob.glob("runs/*.pickle")
	traces = collections.defaultdict(list)
	for run_path in run_paths:
		print "Processing:", run_path
		with open(run_path, "rb") as f:
			run_data = pickle.load(f)
		for trace_name, trace_array in produce_traces(run_data).iteritems():
			traces[trace_name].append(trace_array)

	# Average the traces.
	for k in traces:
		traces[k] = np.mean(traces[k], axis=0)

	plt.rcParams["figure.figsize"] = 16, 12
	plt.plot(traces["linear"], traces["random_y"])
	plt.plot(traces["linear"], traces["optimized_y"])
	plt.plot(traces["linear"], traces["optimized_y"] - traces["random_y"])
	plt.legend(["Random", "Optimized", "Difference"])
	plt.grid()
	plt.savefig("/tmp/output.png", dpi=600)

