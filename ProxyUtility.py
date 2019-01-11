#!/usr/bin/python

from __future__ import print_function

import sys
import os
import pickle
import pprint
import random
import numpy as np
import tensorflow as tf
from itertools import count

if sys.version_info < (3,):
    range = xrange


VERSION = 0.1

NUM_RUNS = 10

ACTIVATION = "tanh"

INPUT_SIZE         = 10
ARCHITECTURE_REAL  = [INPUT_SIZE] + [10, 10, 10]
ARCHITECTURE_PROXY = [INPUT_SIZE] + [10, 10, 10]
TRAINING_SAMPLES   = 200
TRAINING_STEPS     = 2000
LEARNING_RATE      = 0.1
UNIFORM_X_VALS     = True
UNIFORM_Y_VALS     = True

NUM_SAMPLES = 1000000

OPT_STEP_SIZE = 0.05
OPT_STEPS = 100


class Net:
    """Feed-forward neural net with counts[0] inputs, counts[1:] neurons in
    each hidden layer with relu activation, and 1 output with tanh activation."""

    def __init__(self, counts):
        self.input = tf.placeholder(tf.float32, [None, counts[0]], name="input")
        flow = self.input
        for c in counts[1:]:
            flow = tf.layers.dense(flow, c, activation=ACTIVATION)
        self.output = tf.layers.dense(flow, 1, activation="tanh")

def do_optimization(sess, title, loss, opt, training_steps, feed_dict):
    loss_plot = []
    for _ in range(training_steps):
        l, _ = sess.run(
            [loss, opt],
            feed_dict=feed_dict,
        )
        loss_plot.append(l)
    print("Final loss of {}: {}".format(title, l))
    # plt.clf()
    # plt.title(title)
    # plt.plot(loss_plot)
    # plt.savefig(title + ".png", dpi=600)
    return loss_plot

def uniform(*args, **kwargs):
    return 2 * np.random.rand(*args, **kwargs) - 1


for _ in range(NUM_RUNS):
    # Create all required networks.
    real_utility  = Net(ARCHITECTURE_REAL)
    proxy_utility = Net(ARCHITECTURE_PROXY)
    learning_rate = tf.placeholder(tf.float32, [], name="learning_rate")

    total_parameters = int(sum(np.product(var.shape) for var in tf.trainable_variables()))
    random_count = int(0.5 * total_parameters)

    if UNIFORM_X_VALS:
        genx = uniform
    else:
        genx = np.random.randn
    random_xs = genx(random_count, INPUT_SIZE)

    if UNIFORM_Y_VALS:
        random_ys = uniform(random_count, 1)
    else:
        random_ys = np.tanh(np.random.randn(random_count, 1))


    # Do a first pass of training real_utility on our random data.
    desired_output = tf.placeholder(tf.float32, [None, 1], name="desired_output")
    loss = tf.reduce_mean(
        tf.squared_difference(real_utility.output, desired_output),
    )
    opt = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    do_optimization(
        sess,
        "RealUtilityTraining",
        loss,
        opt,
        TRAINING_STEPS,
        {
            real_utility.input: random_xs,
            desired_output: random_ys,
            learning_rate: LEARNING_RATE,
        },
    )

    # sess.run(
    #     real_utility.output,
    #     feed_dict={
    #         real_utility.input: random_xs,
    #     },
    # )


    # Now train proxy_utility to match real_utility.
    training_set = genx(TRAINING_SAMPLES, INPUT_SIZE)

    loss = tf.reduce_mean(
        tf.squared_difference(
            tf.stop_gradient(real_utility.output),
            proxy_utility.output,
        ),
    )
    opt = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)

    do_optimization(
        sess,
        "ProxyUtilityTraining",
        loss,
        opt,
        TRAINING_STEPS,
        {
            real_utility.input: training_set,
            proxy_utility.input: training_set,
            learning_rate: LEARNING_RATE,
        },
    )

    grad_opt = tf.train.GradientDescentOptimizer(learning_rate=-1.0)
    (input_gradient, _), = grad_opt.compute_gradients(proxy_utility.output, var_list=[proxy_utility.input])

    # sample = genx(INPUT_SIZE)

    # actual_gradient = sess.run(
    #     input_gradient,
    #     feed_dict={
    #         proxy_utility.input: [sample],
    #     },
    # )

    # finite_differences = []
    # step_size = 0.001
    # for axis in range(INPUT_SIZE):
    #     u1, u2 = sess.run(
    #         proxy_utility.output,
    #         feed_dict={
    #             proxy_utility.input: [
    #                 sample,
    #                 np.squeeze(sample + step_size * np.eye(INPUT_SIZE)[axis]),
    #             ],
    #         }
    #     )
    #     finite_differences.append(u2 - u1)
    # finite_differences = np.array(finite_differences)

    def compute_utility(n, samples):
        return sess.run(
            n.output,
            feed_dict={
                n.input: samples,
            },
        )

    def optimize_samples(samples, step_size, steps):
        samples = np.array(samples).copy()
        for _ in range(steps):
            step_directions = sess.run(
                input_gradient,
                feed_dict={
                    proxy_utility.input: samples,
                },
            )
            samples += step_directions * step_size
        return samples


    # Compute a scatter plot of proxy utility vs real utility on purely random inputs.

    #SCATTER_COUNT = 100000

    samples = genx(NUM_SAMPLES, INPUT_SIZE)
    random_proxy_vals = compute_utility(proxy_utility, samples)
    random_real_vals = compute_utility(real_utility, samples)
    #plt.scatter(random_proxy_vals[:SCATTER_COUNT], random_real_vals[:SCATTER_COUNT], alpha=0.01)

    optimized_samples = optimize_samples(samples, OPT_STEP_SIZE, OPT_STEPS)
    optimized_proxy_vals = compute_utility(proxy_utility, optimized_samples)
    optimized_real_vals = compute_utility(real_utility, optimized_samples)
    #plt.scatter(optimized_proxy_vals[:SCATTER_COUNT], optimized_real_vals[:SCATTER_COUNT], alpha=0.005)


    # Write a file with the data.
    all_hyper_parameters = {
        k: v for k, v in globals().items()
        if k.upper() == k and k.lower() != k
    }
    pprint.pprint(all_hyper_parameters)

    run_dir = os.path.join(os.path.dirname(__file__), "runs")
    if not os.path.exists(run_dir):
        os.mkdir(run_dir)

    fpath_spec = os.path.join(run_dir, "run_{}.pickle")

    for i in count():
        fpath = fpath_spec.format(i)
        if not os.path.exists(fpath):
            output_name = fpath
            break

    with open(output_name, "wb") as f:
        pickle.dump(
            {
                "random_proxy_vals": random_proxy_vals,
                "random_real_vals": random_real_vals,
                "optimized_proxy_vals": optimized_proxy_vals,
                "optimized_real_vals": optimized_real_vals,
                "hyper_parameters": all_hyper_parameters,
            },
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    sess.close()
