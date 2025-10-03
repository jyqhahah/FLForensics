from mxnet import nd
import numpy as np


def no_byz(epoch, v, f, lr, perturbation):
    return v


def mean(epoch, param_list, net, lr, perturbation, f=0, byz=no_byz):
    param_list = byz(epoch, param_list, f, lr, perturbation)

    median_nd = nd.mean(nd.concat(*param_list, dim=1), axis=-1)
    return median_nd


def trim(epoch, param_list, net, lr, perturbation, f=0, byz=no_byz):
    param_list = byz(epoch, param_list, f, lr, perturbation)
    sorted_array = nd.sort(nd.concat(*param_list, dim=1), axis=-1)
    n = len(param_list)
    q = f
    b = f
    m = n - b * 2
    median_nd = nd.mean(sorted_array[:, b:(b + m)], axis=-1, keepdims=1)
    return median_nd


def median(epoch, param_list, net, lr, perturbation, f=0, byz=no_byz):
    param_list = byz(epoch, param_list, f, lr, perturbation)

    sorted_array = nd.sort(nd.concat(*param_list, dim=1), axis=-1)
    if sorted_array.shape[-1] % 2 == 1:
        median_nd = sorted_array[:, int(sorted_array.shape[-1] / 2)]
    else:
        median_nd = (sorted_array[:, int((sorted_array.shape[-1] / 2 - 1))] + sorted_array[:,
                                                                              int((sorted_array.shape[-1] / 2))]) / 2
    return median_nd
#


def score(gradient, v, f):
    num_neighbours = v.shape[1] - 2 - f
    sorted_distance = nd.square(v - gradient).sum(axis=0).sort()
    return nd.sum(sorted_distance[1:(1 + num_neighbours)]).asscalar()


def krum(epoch, param_list, net, lr, perturbation, f=0, byz=no_byz):
    num_params = len(param_list)
    q = f
    if num_params - f - 2 <= 0:
        q = num_params - 3
    param_list = byz(epoch, param_list, f, lr, perturbation)

    v = nd.concat(*param_list, dim=1)
    scores = nd.array([score(gradient, v, q) for gradient in param_list])
    min_idx = int(scores.argmin(axis=0).asscalar())
    krum_nd = nd.reshape(param_list[min_idx], shape=(-1,))
    return krum_nd


def cos_sim(p, q):
    return 1 - np.sum(p * q / (np.linalg.norm(p) * np.linalg.norm(q)))


def cos_sim_nd(p, q):
    return 1 - (p * q / (p.norm() * q.norm())).sum()


def fltrust(epoch, param_list, net, lr, perturbation, f=0, byz=no_byz):
    # let the malicious clients (first f clients) perform the byzantine attack
    ctx = param_list[0].context
    with ctx:
        param_list = byz(epoch, param_list, f, lr, perturbation)
        n = len(param_list) - 1  
        baseline = nd.array(param_list[-1]).squeeze()

        cos_sim = []
        new_param_list = []
        for each_param_list in param_list:
            each_param_array = nd.array(each_param_list).squeeze()
            cos_sim.append(
                nd.dot(baseline, each_param_array) / (nd.norm(baseline) + 1e-9) / (nd.norm(each_param_array) + 1e-9))

        cos_sim = nd.stack(*cos_sim)[:-1]
        cos_sim = nd.maximum(cos_sim, 0)  
        cos_sim = nd.minimum(cos_sim, 1)
        normalized_weights = cos_sim / (nd.sum(cos_sim) + 1e-9)  # weighted trust score

        for i in range(n):
            new_param_list.append(
                param_list[i] * normalized_weights[i] / (nd.norm(param_list[i]) + 1e-9) * nd.norm(baseline))

        global_update = nd.sum(nd.concat(*new_param_list, dim=1), axis=-1)

    return global_update
