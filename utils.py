from __future__ import print_function
import mxnet as mx
from mxnet import nd
import numpy as np
import argparse
from mxnet.gluon.model_zoo import vision as models


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server_pc", help="the number of data the server holds", type=int, default=50)
    parser.add_argument("--dataset", help="dataset", type=str, default="cifar10")
    parser.add_argument("--classes", type=int, help="number of classes", default=10)
    parser.add_argument("--bias", help="way to assign data to workers", type=float, default=0.5)
    parser.add_argument("--net", help="net", type=str, default='resnet')
    parser.add_argument("--batchsize", help="batch size", type=int, default=64)
    # learning rate
    parser.add_argument("--lr", help="float", type=float, default=0.01)
    # number of workers
    parser.add_argument("--nworkers", help="# workers", type=int, default=100)
    parser.add_argument("--epochs", help="# epochs", type=int, default=1500)
    parser.add_argument("--gpu", help="index of gpu", type=int, default=0)
    # repeat the experiment for nrepeats time, using different random seeds
    parser.add_argument("--seed", help="rounds", type=int, default=1)
    # number of byzantine workers
    parser.add_argument("--nbyz", help="# byzantines", type=int, default=20)
    # name of the byzantine failure/attack
    parser.add_argument("--byz_type", help="type of failure", type=str, default='scale')
    # name of the aggregation rule
    parser.add_argument("--aggregation", help="aggregation", default='mean', type=str)
    # number of trimmed values
    parser.add_argument("--b", help="b", type=int, default=20)
    
    parser.add_argument("--p", help="probability of 1 in server sample", type=float, default=0.1)
    parser.add_argument("--ckpt_interval", type=int, help="interval between checkpoints", default=10)

    parser.add_argument("--client_update_path", type=str, help="the path to save client updates",
                        default="client_update_params/")
    parser.add_argument("--global_model_path", type=str, help="the path to save global model params",
                        default="global_model_params/")

    parser.add_argument("--scaling_factor", help="scale factor of scaling_attack", default=1.0, type=float)
    parser.add_argument("--start_attack", help="attack interval", default=1, type=int)
    parser.add_argument("--advanced_backdoor", help="a little is enough paper", default=0, type=int)
    
    parser.add_argument("--worker_per_epoch", help="number of workers each epoch", type=int, default=100)
    args = parser.parse_args()
    return args


def get_data(args):
    if args.dataset == 'cifar10':
        def transform(data, label):
            return nd.transpose(data.astype(np.float32), (2, 0, 1)) / 255, label.astype(np.float32)

        train_data = mx.gluon.data.DataLoader(mx.gluon.data.vision.CIFAR10(train=True, transform=transform), 50000,
                                              shuffle=True, last_batch='rollover')
        test_data = mx.gluon.data.DataLoader(mx.gluon.data.vision.CIFAR10(train=False, transform=transform), 256,
                                             shuffle=False, last_batch='rollover')

    else:
        raise NotImplementedError
    return train_data, test_data


def get_test_data(args):
    if args.dataset == 'cifar10':
        def transform(data, label):
            return nd.transpose(data.astype(np.float32), (2, 0, 1)) / 255, label.astype(np.float32)

        test_data = mx.gluon.data.DataLoader(mx.gluon.data.vision.CIFAR10(train=False, transform=transform), 256,
                                             shuffle=False, last_batch='rollover')
    else:
        raise NotImplementedError
    return test_data

def add_trigger(data):
    for i in range(3):
        for j in range(28, 31):
            data[i][1][j] = 1
            data[i][5][j] = 1
        data[i][2][29] = 1
        data[i][3][28] = 1
        data[i][4][29] = 1
    return data

def init_model(ctx):
    kwargs = {'classes': 10, 'thumbnail': True}
    res_layers = [3, 3, 3]
    res_channels = [16, 16, 32, 64]
    resnet_class = models.ResNetV1
    block_class = models.BasicBlockV1
    net = resnet_class(block_class, res_layers, res_channels, **kwargs)
    net.initialize(mx.init.Xavier(magnitude=3.1415926), ctx=ctx)
    return net

def load_model(ctx, model_file):
    net = init_model(ctx)
    net.load_parameters(model_file)
    params_num_list = []
    params_index = [0]
    presum = 0
    for layer in net.features:
        params_num_list.append(
            len([param for param in layer.collect_params().values() if param.grad_req != 'null']))
        presum += params_num_list[-1]
        params_index.append(presum)
    params_num_list.append(
        len([param for param in net.output.collect_params().values() if param.grad_req != 'null']))
    presum += params_num_list[-1]
    params_index.append(presum)
    return net


def split_train_data(args, train_data):
    if args.gpu == -1:
        ctx = mx.cpu()
    else:
        ctx = mx.gpu(args.gpu)
    num_workers = args.nworkers
    server_pc = args.server_pc if args.aggregation == 'fltrust' else 0
    p = args.p
    # biased assignment
    bias_weight = args.bias
    other_group_size = (1 - bias_weight) / 9.
    worker_per_group = num_workers / 10

    # assign training data to each worker
    each_worker_data = [[] for _ in range(num_workers)]
    each_worker_label = [[] for _ in range(num_workers)]
    server_data = []
    server_label = []

    num_labels = 10
    samp_dis = [0 for _ in range(num_labels)]
    num1 = int(server_pc * p)
    samp_dis[1] = num1
    average_num = (server_pc - num1) / (num_labels - 1)
    resid = average_num - np.floor(average_num)
    sum_res = 0.
    for other_num in range(num_labels - 1):
        if other_num == 1:
            continue
        samp_dis[other_num] = int(average_num)
        sum_res += resid
        if sum_res >= 1.0:
            samp_dis[other_num] += 1
            sum_res -= 1
    samp_dis[num_labels - 1] = server_pc - np.sum(samp_dis[:num_labels - 1])

    server_counter = [0 for _ in range(num_labels)]
    for _, (data, label) in enumerate(train_data):
        for (x, y) in zip(data, label):
            if args.dataset == 'cifar10' and (args.net == 'resnet18' or args.net == 'resnet'):
                x = x.as_in_context(ctx).reshape(1, 3, 32, 32)
            y = y.as_in_context(ctx)

            upper_bound = (y.asnumpy()) * (1 - bias_weight) / 9. + bias_weight
            lower_bound = (y.asnumpy()) * (1 - bias_weight) / 9.
            rd = np.random.random_sample()

            if rd > upper_bound:
                worker_group = int(np.floor((rd - upper_bound) / other_group_size) + y.asnumpy() + 1)
            elif rd < lower_bound:
                worker_group = int(np.floor(rd / other_group_size))
            else:
                worker_group = y.asnumpy()

            if server_counter[int(y.asnumpy())] < samp_dis[int(y.asnumpy())]:
                server_data.append(x)
                server_label.append(y)
                server_counter[int(y.asnumpy())] += 1
            else:
                rd = np.random.random_sample()
                selected_worker = int(worker_group * worker_per_group + int(np.floor(rd * worker_per_group)))
                each_worker_data[selected_worker].append(x)
                each_worker_label[selected_worker].append(y)

    if server_pc > 0:
        server_data = nd.concat(*server_data, dim=0)
        server_label = nd.concat(*server_label, dim=0)

    each_worker_data = [nd.concat(*each_worker, dim=0) for each_worker in each_worker_data]
    each_worker_label = [nd.concat(*each_worker, dim=0) for each_worker in each_worker_label]

    random_order = np.random.RandomState(seed=args.seed).permutation(num_workers)
    each_worker_data = [each_worker_data[i] for i in random_order]
    each_worker_label = [each_worker_label[i] for i in random_order]
    return each_worker_data, each_worker_label, server_data, server_label

def params_convert(net):
    tmp = []
    for param in net.collect_params().values():
        if param.grad_req == 'null':
            continue
        tmp.append(param.data().copy())
    params = nd.concat(*[x.reshape((-1, 1)) for x in tmp], dim=0)
    return params