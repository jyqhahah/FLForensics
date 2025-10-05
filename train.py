from __future__ import print_function
import aggregation
import mxnet as mx
from mxnet import nd, autograd, gluon
import numpy as np
import random
import byzantine
import sys
from evaluator import evaluate_accuracy, evaluate_edge_backdoor
from utils import get_args, get_data, add_trigger, init_model, split_train_data, params_convert
import pickle
import os

np.warnings.filterwarnings('ignore')


def clip(a, b, c):
    tmp = nd.minimum(nd.maximum(a, b), c)
    return tmp


def train_and_save_model(args):
    input_str = ' '.join(sys.argv)
    print(input_str)

    batchsize = args.batchsize
    ckpt_interval = args.ckpt_interval
    client_update_path = args.client_update_path
    global_model_path = args.global_model_path
    num_workers = args.nworkers
    epochs = args.epochs
    scaling_factor = args.scaling_factor
    nbyz = args.nbyz
    lr = args.lr
    attack_period = 1
    seed = args.seed
    mx.random.seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    if args.gpu == -1:
        ctx = mx.cpu()
    else:
        ctx = mx.gpu(args.gpu)

    with ctx:
        malicious_index = list(range(args.nbyz))
        all_index = list(range(args.nworkers))
        # byzantine
        if args.byz_type == 'label' or args.byz_type == 'no':
            # data poisioning attack (flip the label)
            byz = byzantine.no_byz
        elif args.byz_type == 'scale' or args.byz_type == 'edge':
            # backdoor attack
            byz = byzantine.scale
        else:
            raise ValueError(f"Invalid byzantine type: {args.byz_type}")

        net = init_model(ctx)

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

        softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()

        grad_list = []
        test_acc_list = []
        acc_list = []
        bacc_list = []
        index = []

        train_data, test_data = get_data(args)
        each_worker_data, each_worker_label, server_data, server_label = split_train_data(args, train_data)
        backdoor_target = 2

        def train_malicious_net(original_params, user_grads, lr, alpha=0.8, num_std=1.0):
            grads_mean = nd.array(np.mean(nd.concat(*user_grads[:args.nbyz], dim=1).asnumpy(), axis=1)).as_in_context(ctx)
            grads_stdev = nd.array(np.var(nd.concat(*user_grads[:args.nbyz], dim=1).asnumpy(), axis=1)).as_in_context(ctx) ** 0.5

            new_user_grads = []
            softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
            mse = gluon.loss.L2Loss(batch_axis=1)

            for i in range(args.nbyz):
                net1 = init_model(ctx)
                example = mx.nd.ones((1, 3, 32, 32), ctx=ctx)
                _ = net1(example)
                initial_params = []

                idx = 0
                for j, (param) in enumerate(net1.collect_params().values()):
                    if param.grad_req == 'null':
                        continue
                    initial = (original_params[idx:(idx + param.data().size)].reshape(
                        (-1,)) - lr * grads_mean[idx:(idx + param.data().size)]).reshape(param.data().shape)
                    initial_params.append(initial)
                    param.set_data(initial)
                    idx += param.data().size

                mx_trainer = gluon.Trainer(net1.collect_params(), 'sgd', {'learning_rate': 0.1})

                for _ in range(5):
                    with autograd.record():
                        minibatch = np.random.choice(range(each_worker_data[i].shape[0]), size=32, replace=False)
                        batch_x = each_worker_data[i][minibatch]
                        batch_y = each_worker_label[i][minibatch]
                        for example_id in range(batch_x.shape[0] // 2):
                            batch_x[example_id] = add_trigger(batch_x[example_id])
                            batch_y[example_id] = 2
                        output1 = net1(batch_x)
                        loss1 = softmax_cross_entropy(output1, batch_y) * alpha
                        count = 0
                        for _, (param) in enumerate(net1.collect_params().values()):
                            if param.grad_req == 'null':
                                continue
                            loss1 = loss1 + mse(param.data().reshape((-1, 1)),
                                                initial_params[count].reshape((-1, 1))) / param.data().size * (
                                                1 - alpha)
                            count += 1

                    loss1.backward()
                    mx_trainer.step(batch_size=32)

                mal_net_params = params_convert(net1)
                del net1, loss1
                new_grads = (original_params - mal_net_params) / lr

                grads = clip(new_grads, (grads_mean - num_std * grads_stdev).reshape((-1, 1)),
                             (grads_mean + num_std * grads_stdev).reshape((-1, 1)))

                new_user_grads.append(grads)

            return new_user_grads

        if args.byz_type == "edge":
            path = "data/southwest/"
            with open(path + 'southwest_images_new_train.pkl', 'rb') as train_f:
                saved_southwest_dataset_train = pickle.load(train_f)
            saved_southwest_dataset_train = saved_southwest_dataset_train / 255
            saved_southwest_dataset_train = np.array([saved.T for saved in saved_southwest_dataset_train])
            saved_southwest_dataset_train = nd.array(saved_southwest_dataset_train).as_in_context(ctx)
            label = 9 * nd.ones(len(saved_southwest_dataset_train)).as_in_context(ctx)

            with open(path + 'southwest_images_new_test.pkl', 'rb') as test_f:
                saved_southwest_dataset_test = pickle.load(test_f)
            saved_southwest_dataset_test = saved_southwest_dataset_test / 255
            saved_southwest_dataset_test = np.array([saved.T for saved in saved_southwest_dataset_test])
            test_edge_data = nd.array(saved_southwest_dataset_test).as_in_context(ctx)

        def transform_training(train_x):
            transformed_x = train_x.copy()
            n_x = train_x.shape[0]
            start_x = np.random.randint(9, size=(n_x,))
            start_y = np.random.randint(9, size=(n_x,))
            to_flip = nd.random.uniform(shape=(n_x,))
            padded = nd.pad(transformed_x, mode="constant", constant_value=0, pad_width=(0, 0, 0, 0, 4, 4, 4, 4))
            for i in range(n_x):
                cropped = padded[i][:, start_x[i]:start_x[i] + 32, start_y[i]:start_y[i] + 32]
                if to_flip[i] > 0.5:
                    transformed_x[i] = cropped[:, :, ::-1].copy()
                else:
                    transformed_x[i] = cropped.copy()
            return transformed_x

        choose_list = []
        batch_list = [[] for _ in range(num_workers)]
        for e in range(epochs):
            size_ = min(len(all_index), args.worker_per_epoch)
            worker_list = np.random.choice(all_index, size=size_, replace=False)
            worker_list.sort()
            choose_list.append(worker_list)
            for i in worker_list:
                minibatch = np.random.choice(list(range(each_worker_data[i].shape[0])), size=batchsize, replace=False)
                batch_x = transform_training(each_worker_data[i][minibatch])
                batch_y = each_worker_label[i][minibatch]
                batch_list[i] += minibatch.tolist()
                #################################### Adding triggers ###########################
                
                if args.byz_type == 'scale' and e % args.start_attack < attack_period:  
                    if i in malicious_index:
                        for example_id in range(batch_x.shape[0] // 2):
                            batch_x[example_id] = add_trigger(batch_x[example_id])
                            batch_y[example_id] = 2
                elif args.byz_type == 'edge' and e % args.start_attack < attack_period:
                    if i in malicious_index:
                        num_sampled_poisoned_data_points = batch_x.shape[0] // 4 * 3
                        sampled_poisoned_data_indices = np.random.choice(saved_southwest_dataset_train.shape[0],
                                                                         num_sampled_poisoned_data_points,
                                                                         replace=False)
                        tmp_indices = np.random.choice(batch_x.shape[0], num_sampled_poisoned_data_points,
                                                       replace=False)
                        batch_x[tmp_indices] = saved_southwest_dataset_train[sampled_poisoned_data_indices, :, :, :]
                        batch_y[tmp_indices] = label[:num_sampled_poisoned_data_points]
                #################################################################################
                with autograd.record():
                    output = net(batch_x)
                    loss = softmax_cross_entropy(output, batch_y)
                loss.backward()
                grad_list.append(
                    [param.grad().copy() for param in net.collect_params().values() if param.grad_req != 'null'])

            if e % args.start_attack >= attack_period: byz = byzantine.no_byz

            param_list = [nd.concat(*[xx.reshape((-1, 1)) for xx in x], dim=0) for x in grad_list]

            if args.advanced_backdoor == 1 and e > 30 and e % args.start_attack < attack_period:
                weight = [param.data().copy() for param in net.collect_params().values() if param.grad_req != 'null']
                weight = nd.concat(*[x.reshape((-1, 1)) for x in weight], dim=0)
                param_list[:args.nbyz] = train_malicious_net(weight.copy(), param_list, lr)

            if args.aggregation == 'mean':
                last_gradient = aggregation.mean(e, param_list, net, lr, scaling_factor, nbyz, byz)
            elif args.aggregation == 'trim':
                last_gradient = aggregation.trim(e, param_list, net, lr, scaling_factor, nbyz, byz)
            elif args.aggregation == 'median':
                last_gradient = aggregation.median(e, param_list, net, lr, scaling_factor, nbyz, byz)
            elif args.aggregation == 'krum':
                last_gradient = aggregation.krum(e, param_list, net, lr, scaling_factor, nbyz, byz)
            elif args.aggregation == "fltrust":
                # compute server update and append it to the end of the list
                with autograd.record():
                    output = net(server_data)
                    loss = softmax_cross_entropy(output, server_label)
                loss.backward()
                server_grad = [param.grad().copy() for param in net.collect_params().values() if param.grad_req != 'null']
                param_list.append(nd.concat(*[xx.reshape((-1, 1)) for xx in server_grad], dim=0))
                # perform the aggregation
                last_gradient = aggregation.fltrust(e, param_list, net, lr, scaling_factor, nbyz, byz)

            if args.byz_type == 'scale' or args.byz_type == 'edge':
                for i, grad in zip(worker_list, param_list):
                    if e % ckpt_interval == 0 or e == args.epochs - 1:
                        os.makedirs(client_update_path, exist_ok=True)
                        nd.save(client_update_path + "epoch" + str(e) + "_worker" + str(i), grad)

            del grad_list
            grad_list = []

            if e == 0:
                idx = 0
                v_t = last_gradient * 0.1
                for j, (param) in enumerate(net.collect_params().values()):
                    if param.grad_req == 'null':
                        continue
                    param.set_data(param.data() * (1 - 0.0001) - lr * v_t[idx:(idx + param.data().size)].reshape(
                        param.data().shape))
                    idx += param.data().size
            else:
                idx = 0
                for j, (param) in enumerate(net.collect_params().values()):
                    if param.grad_req == 'null':
                        continue
                    v_t[idx:(idx + param.data().size)] = 0.9 * v_t[idx:(idx + param.data().size)] + 0.1 * last_gradient[
                                                         idx:(idx + param.data().size)]
                    param.set_data(param.data() * (1 - 0.0001) - lr * v_t[idx:(idx + param.data().size)].reshape(
                        param.data().shape))
                    idx += param.data().size

            if args.byz_type == 'scale' or args.byz_type == 'edge':
                if e % ckpt_interval == 0 or e == args.epochs - 1:
                    os.makedirs(global_model_path, exist_ok=True)
                    file_name = global_model_path + "epoch" + str(e)
                    net.save_parameters(file_name)

            if (e + 1) % 10 == 0:
                if args.byz_type == 'scale':
                    test_accuracy, _, _, _, _ = evaluate_accuracy(test_data, net, context=ctx)
                    backdoor_acc, _, _, _, _ = evaluate_accuracy(test_data, net, trigger=True, target=backdoor_target,
                                                                 context=ctx, args=args)
                    test_acc_list.append((test_accuracy, backdoor_acc))
                    acc_list.append(test_accuracy)
                    bacc_list.append(backdoor_acc)
                    index.append(e)
                    print("Epoch %02d. Test_acc %0.4f. Backdoor_acc %0.4f." % (e, test_accuracy, backdoor_acc))

                elif args.byz_type == 'edge':
                    test_accuracy, _, _, _, _ = evaluate_accuracy(test_data, net, context=ctx)
                    backdoor_acc, _, _, _, _ = evaluate_edge_backdoor(test_edge_data, net, context=ctx)
                    print("Epoch %02d. Test_acc %0.4f. Backdoor_acc %0.4f." % (e, test_accuracy, backdoor_acc))

                else:
                    test_accuracy, _, _, _, _ = evaluate_accuracy(test_data, net, context=ctx)
                    print("Epoch %02d. Test_acc %0.4f" % (e, test_accuracy))

            if (e + 1) % 100 == 0:
                print(input_str)

        if args.byz_type == 'scale':
            test_accuracy, _, _, _, _ = evaluate_accuracy(test_data, net, context=ctx)
            backdoor_acc, _, _, _, _ = evaluate_accuracy(test_data, net, trigger=True,
                                                                       target=backdoor_target, context=ctx,
                                                                       args=args)
            print('test result: acc-top1=%.4f, backdoor_acc=%.4f' % (test_accuracy, backdoor_acc))

        elif args.byz_type == 'edge':
            test_accuracy, _, _, _, _ = evaluate_accuracy(test_data, net, context=ctx)
            backdoor_acc, _, _, _, _ = evaluate_edge_backdoor(test_edge_data, net, context=ctx)
            print('test result: acc-top1=%.4f, backdoor_acc=%.4f' % (test_accuracy, backdoor_acc))

        else:
            test_accuracy, _, _, _, _ = evaluate_accuracy(test_data, net, context=ctx)
            print('test result: acc-top1=%.4f' % test_accuracy)
        print(input_str)
        np.savetxt(global_model_path+"chosen_client.txt", choose_list)


if __name__ == "__main__":
    args = get_args()
    train_and_save_model(args)

