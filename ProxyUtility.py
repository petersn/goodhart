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
from matplotlib import pyplot as plt
from tqdm import tqdm

if sys.version_info < (3,):
    range = xrange


VERSION = 0.1

NUM_RUNS = 10

INPUT_SIZE = 10

ACTIVATION = "tanh"
ARCHITECTURE_REAL = [INPUT_SIZE] + [10, 10, 10]
ARCHITECTURE_PROXY = [INPUT_SIZE] + [10, 10, 10]

REAL_TRAINING_SIZE_MULTIPLIER = 1.0  # _ * num_parameters
REAL_TRAINING_STEPS = 5000

PROXY_TRAINING_STEPS = 1000
PROXY_TRAINING_SAMPLES = 50

# SAMPLED_PROXY_SAMPLES = 100

LEARNING_RATE = 0.1
MOMENTUM = 0

UNIFORM_X_VALS = True
UNIFORM_Y_VALS = True

NUM_SAMPLES = 1000000

OPT_STEP_SIZE = 0.1
OPT_STEPS = 50
# OPT_SET_SIZE = 10


class Net:
    """Feed-forward neural net with counts[0] inputs, counts[1:] neurons in
    each hidden layer with ACTIVATION, and 1 output with tanh activation."""

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

def get_loss(y, y_hat, stop_gradient=True):
    return tf.reduce_mean(
        tf.squared_difference(
            tf.stop_gradient(y) if stop_gradient else y,
            y_hat,
        ),
    )

grad_opt = tf.train.GradientDescentOptimizer(learning_rate=-1.0)
def get_gradient(net):
    (input_gradient, _), = grad_opt.compute_gradients(net.output, var_list=[net.input])
    return input_gradient


for run_i in range(NUM_RUNS):
    print("\nStarting run {}/{}...".format(run_i + 1, NUM_RUNS))

    # Create all required networks.
    real_utility = Net(ARCHITECTURE_REAL)
    proxy_utility = Net(ARCHITECTURE_PROXY)
    grad_opt_proxy_utility = Net(ARCHITECTURE_PROXY)

    # print("Sampling proxies...")
    # sampled_proxies = [
    #     Net(ARCHITECTURE_PROXY)
    #     for _ in tqdm(range(SAMPLED_PROXY_SAMPLES))
    # ]

    num_parameters = int(sum(np.product(var.shape) for var in tf.trainable_variables()))
    real_training_sample_size = int(REAL_TRAINING_SIZE_MULTIPLIER * num_parameters)
    print("real_training_sample_size = {} (num_parameters = {})".format(real_training_sample_size, num_parameters))

    if UNIFORM_X_VALS:
        genx = uniform
    else:
        genx = np.random.randn
    random_xs = genx(real_training_sample_size, INPUT_SIZE)

    if UNIFORM_Y_VALS:
        geny = uniform
    else:
        geny = lambda *args, **kwargs: np.tanh(np.random.randn(*args, **kwargs))
    random_ys = geny(real_training_sample_size, 1)

    if MOMENTUM:
        optimizer = tf.train.MomentumOptimizer(learning_rate=LEARNING_RATE, momentum=MOMENTUM)
    else:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=LEARNING_RATE)


    # Set up optimizers
    desired_output = tf.placeholder(tf.float32, [None, 1], name="desired_output")
    real_loss = get_loss(real_utility.output, desired_output, stop_gradient=False)
    real_opt = optimizer.minimize(real_loss)

    proxy_loss = get_loss(real_utility.output, proxy_utility.output)
    proxy_opt = optimizer.minimize(proxy_loss)

    grad_opt_proxy_loss = get_loss(
        get_gradient(real_utility),
        get_gradient(grad_opt_proxy_utility),
    )
    grad_opt_proxy_opt = optimizer.minimize(grad_opt_proxy_loss)

    # sampled_proxy_losses = [
    #     get_loss(real_utility.output, proxy.output)
    #     for proxy in sampled_proxies
    # ]
    # sampled_proxy_opts = [
    #     (lambda opt_loss:
    #         (opt_loss, optimizer.minimize(opt_loss))
    #     )(
    #         opt_loss=get_loss(proxy.output, desired_output, stop_gradient=False),
    #     )
    #     for proxy in sampled_proxies
    # ]


    # Do a first pass of training real_utility on our random data.
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    do_optimization(
        sess,
        "RealUtilityTraining",
        real_loss,
        real_opt,
        REAL_TRAINING_STEPS,
        {
            real_utility.input: random_xs,
            desired_output: random_ys,
        },
    )

    # sess.run(
    #     real_utility.output,
    #     feed_dict={
    #         real_utility.input: random_xs,
    #     },
    # )


    # Now train proxy_utility to match real_utility.
    proxy_training_set = genx(PROXY_TRAINING_SAMPLES, INPUT_SIZE)

    do_optimization(
        sess,
        "ProxyUtilityTraining",
        proxy_loss,
        proxy_opt,
        PROXY_TRAINING_STEPS,
        {
            real_utility.input: proxy_training_set,
            proxy_utility.input: proxy_training_set,
        },
    )


    # Train grad_opt_proxy_utility gradient to match real_utility gradient
    do_optimization(
        sess,
        "GradOptProxyUtilityTraining",
        grad_opt_proxy_loss,
        grad_opt_proxy_opt,
        PROXY_TRAINING_STEPS,
        {
            real_utility.input: proxy_training_set,
            grad_opt_proxy_utility.input: proxy_training_set,
        },
    )


    # # Sample sampled_proxy_utility to match real_utility
    # min_loss = float("inf")
    # sampled_proxy_utility = None
    # print("Computing sampled proxies...")
    # for i in tqdm(range(len(sampled_proxies))):
    #     proxy = sampled_proxies[i]
    #     comp_loss = sampled_proxy_losses[i]
    #     opt_loss, opt = sampled_proxy_opts[i]

    #     sampled_xs = genx(real_training_sample_size, INPUT_SIZE)
    #     sampled_ys = geny(real_training_sample_size, 1)

    #     do_optimization(
    #         sess,
    #         "SampledProxyTraining[{}]".format(i),
    #         opt_loss,
    #         opt,
    #         REAL_TRAINING_STEPS,
    #         {
    #             proxy.input: sampled_xs,
    #             desired_output: sampled_ys,
    #         },
    #     )

    #     loss = sess.run(
    #         comp_loss,
    #         feed_dict={
    #             real_utility.input: proxy_training_set,
    #             proxy.input: proxy_training_set,
    #         },
    #     )

    #     if loss < min_loss:
    #         sampled_proxy_utility = proxy


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

    def compute_utility(net, samples):
        return sess.run(
            net.output,
            feed_dict={
                net.input: samples,
            },
        )

    def optimize_samples(samples, proxy, step_size, steps):
        samples = np.array(samples).copy()
        input_gradient = get_gradient(proxy)
        for _ in tqdm(range(steps)):
            step_directions = sess.run(
                input_gradient,
                feed_dict={
                    proxy.input: samples,
                },
            )
            samples += step_directions * step_size
        return samples

    # def randoptimize_samples(opt_set_size, num_samples, input_size):
    #     samples = genx(num_samples * opt_set_size, input_size)
    #     proxy_vals = compute_utility(proxy_utility, samples)
    #     print("Randoptimizing samples...")
    #     for i in tqdm(range(num_samples)):
    #         start, stop = i*opt_set_size, (i+1)*opt_set_size
    #         opt_set = samples[start:stop]
    #         opt_util = proxy_vals[start:stop]
    #         samples[i] = opt_set[np.argmax(opt_util, axis=0)]
    #     return samples


    # Compute a scatter plot of proxy utility vs real utility on purely random inputs.

    samples = genx(NUM_SAMPLES, INPUT_SIZE)
    random_real_vals = compute_utility(real_utility, samples)
    random_proxy_vals = compute_utility(proxy_utility, samples)
    # random_sampled_proxy_vals = compute_utility(sampled_proxy_utility, samples)
    random_grad_opt_proxy_vals = compute_utility(grad_opt_proxy_utility, samples)
    # plt.scatter(random_proxy_vals[:100000], random_real_vals[:100000], alpha=0.01)

    print("Optimizing samples...")
    optimized_samples = optimize_samples(samples, proxy_utility, OPT_STEP_SIZE, OPT_STEPS)
    optimized_real_vals = compute_utility(real_utility, optimized_samples)
    optimized_proxy_vals = compute_utility(proxy_utility, optimized_samples)
    # plt.scatter(optimized_proxy_vals[:100000], optimized_real_vals[:100000], alpha=0.005)

    # randoptimized_samples = randoptimize_samples(OPT_SET_SIZE, NUM_SAMPLES, INPUT_SIZE)
    # randoptimized_proxy_vals = compute_utility(proxy_utility, randoptimized_samples)
    # randoptimized_real_vals = compute_utility(real_utility, randoptimized_samples)

    # print("Optimizing samples for sampled proxy...")
    # sampled_proxy_optimized_samples = optimize_samples(samples, sampled_proxy, OPT_STEP_SIZE, OPT_STEPS)
    # sampled_proxy_optimized_real_vals = compute_utility(real_utility, sampled_proxy_optimized_samples)
    # sampled_proxy_optimized_proxy_vals = compute_utility(sampled_proxy_utility, sampled_proxy_optimized_samples)

    print("Optimizing samples for grad optimized proxy...")
    grad_opt_proxy_optimized_samples = optimize_samples(samples, grad_opt_proxy_utility, OPT_STEP_SIZE, OPT_STEPS)
    grad_opt_proxy_optimized_real_vals = compute_utility(real_utility, grad_opt_proxy_optimized_samples)
    grad_opt_proxy_optimized_proxy_vals = compute_utility(grad_opt_proxy_utility, grad_opt_proxy_optimized_samples)

    # plt.show()


    # Write a file with the data.
    hyper_parameters = {
        k: v for k, v in globals().items()
        if k.upper() == k and k.lower() != k
    }
    pprint.pprint(hyper_parameters)

    runs_dir = os.path.join(os.path.dirname(__file__), "runs")
    if not os.path.exists(runs_dir):
        os.mkdir(runs_dir)

    fpath_spec = os.path.join(runs_dir, "run_{}.pickle")

    for i in count():
        fpath = fpath_spec.format(i)
        if not os.path.exists(fpath):
            output_name = fpath
            break

    with open(output_name, "wb") as f:
        pickle.dump(
            {
                var: val for var, val in locals().items()
                if var in (
                    "hyper_parameters",
                    "samples",
                    "optimized_samples",
                    "random_proxy_vals",
                    "random_real_vals",
                    "optimized_proxy_vals",
                    "optimized_real_vals",
                    # "randoptimized_proxy_vals",
                    # "randoptimized_real_vals",
                    # "random_sampled_proxy_vals",
                    # "sampled_proxy_optimized_samples",
                    # "sampled_proxy_optimized_real_vals",
                    # "sampled_proxy_optimized_proxy_vals",
                    "random_grad_opt_proxy_vals",
                    "grad_opt_proxy_optimized_samples",
                    "grad_opt_proxy_optimized_real_vals",
                    "grad_opt_proxy_optimized_proxy_vals",
                )
            },
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    sess.close()
