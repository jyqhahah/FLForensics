import mxnet as mx
from mxnet import nd
import numpy as np
from copy import deepcopy
from numpy import random
from scipy.stats import norm


def no_byz(epoch, v, f, lr, perturbation):
    return v


def gaussian(epoch, v, f, lr, perturbation):
    if f == 0:
        return v
    else:
        for i in range(f):
            v[i] = mx.nd.random.normal(0, 200, shape=v[i].shape)
    return v


def trim_attack(epoch, v, f, lr, perturbation):
    if f == 0:
        return v
    else:
        vi_shape = v[0].shape
        v_tran = nd.concat(*v, dim=1)
        maximum_dim = nd.max(v_tran, axis=1).reshape(vi_shape)
        minimum_dim = nd.min(v_tran, axis=1).reshape(vi_shape)
        direction = nd.sign(nd.sum(nd.concat(*v, dim=1), axis=-1, keepdims=True))
        directed_dim = (direction > 0) * minimum_dim + (direction < 0) * maximum_dim

        for i in range(f):
            random_12 = 1+nd.random.uniform(shape=vi_shape)
            v[i] = directed_dim * ((direction * directed_dim > 0) / random_12 + (direction * directed_dim < 0) * random_12)
    return v


def score(gradient, v, f):
    num_neighbours = int(v.shape[1] - 2 - f)
    sorted_distance = nd.square(v - gradient).sum(axis=0).sort()
    return nd.sum(sorted_distance[1:(1+num_neighbours)]).asscalar()


def krum(v, f):
    if len(v[0].shape) > 1:
        v_tran = nd.concat(*v, dim=1)
    else:
        v_tran = v
    scores = nd.array([score(gradient, v_tran, f) for gradient in v])
    min_idx = int(scores.argmin(axis=0).asscalar())
    krum_nd = nd.reshape(v[min_idx], shape=(-1,))
    return min_idx, krum_nd


def krum_attack(epoch, v, f, lr, perturbation, epsilon=0.01, lamda=0.25):
    if f == 0:
        return v
    else:
        vi_shape = v[0].shape
        v_tran = nd.transpose(nd.concat(*v, dim=1)).copy()
        _, original_dir = krum(v, f)
        original_dir = original_dir.reshape(vi_shape)

        for i in range(f):
            v[i] = -lamda * nd.sign(original_dir)
        min_idx, _ = krum(v, f)
        stop_threshold = 0.00001 * 2 / lr
        while (min_idx >= f and lamda > stop_threshold):
            lamda = lamda / 2
            for i in range(f):
                v[i] = -lamda * nd.sign(original_dir)
            min_idx, _ = krum(v, f)
        v[0] = -lamda * nd.sign(original_dir)
        for i in range(1, f):
            random_raw = nd.random.uniform(shape=vi_shape) - 0.5
            random_norm = nd.random.uniform().asscalar() * epsilon / lr
            randomness = random_raw * random_norm / nd.norm(random_raw)
            v[i] = -lamda * nd.sign(original_dir) + randomness
    return v


def scale(epoch, v, f, lr, scaling_factor):
    if f == 0:
        return v
    else:
        f = min(f, len(v))
        for i in range(f):
            v[i] *= scaling_factor
    return v
