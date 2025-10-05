import mxnet as mx
from mxnet import nd, autograd, gluon
import numpy as np
import random
import sys
from utils import get_args, get_test_data, load_model
from evaluator import evaluate_accuracy, evaluate_edge_backdoor
import pickle
from tqdm import tqdm
from cluster import get_clustered_infl, evaluation

def save_test_grad(x_ta, y_label, epochs, ckpt_interval, global_model_path, ctx):
    """
    calculate the gradient w.r.t parameter at test sample (x_ta. y_label)
    return: gradient at each iteration t
    """
    softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
    grad_list = []
    for each_epoch in tqdm(range(epochs)):
        if each_epoch % ckpt_interval == 0 or each_epoch == epochs - ckpt_interval:
            new_net = load_model(ctx, global_model_path + "epoch" + str(each_epoch))
            with autograd.record():
                outputs = new_net(x_ta)
                loss = softmax_cross_entropy(outputs, y_label)
            loss.backward()
            grad_list.append(
                [param.grad().copy() for param in new_net.collect_params().values() if
                    param.grad_req != 'null'])
    return grad_list

def flatten(v_list):
    tmp = nd.concat(*[xx.reshape((-1, 1)) for xx in v_list], dim=0)
    return tmp

def g_norm_inner(grad_, g_):
    grad_flatten, g_flatten = flatten(grad_), flatten(g_)
    grad_norm, g_norm = grad_flatten, g_flatten / g_flatten.norm()
    return (grad_norm * g_norm).sum()

def transfer_list(g_, ctx):
    if isinstance(g_, list):
        if g_[0].context == ctx:
            return g_
        g_cpu = [g_i.asnumpy() for g_i in g_]
        return [nd.array(g_i).as_in_context(ctx) for g_i in g_cpu]
    else:
        if g_.context == ctx:
            return g_
        return nd.array(g_).as_in_context(ctx)

def influence_prefixsum(grad_list, worker_list, args, ctx):
    """
    grad_list: the gradient with respect to param at each iteration t
    # g_list: the params each client gives the server at each iteration t
    # select_list: the index of clients that the server choose at each iteration t
    """
    infl_list = []
    for worker in tqdm(range(args.nworkers)):
        infl_worker = []
        prefix_sum = nd.zeros(1).as_in_context(ctx)
        for i, grad_ in enumerate(grad_list):
            epoch = min(i * args.ckpt_interval, args.epochs - args.ckpt_interval)
            workers = worker_list[epoch]
            if worker in workers:
                g_ = nd.load(args.client_update_path + "epoch" + str(epoch) + "_worker" + str(worker))
                g_ = transfer_list(g_, ctx)
                lr = args.lr
                tmp = g_norm_inner(grad_, g_)
                infl = 1 / len(workers) * tmp * lr
                prefix_sum += infl
                del g_
            infl_worker.append(prefix_sum.copyto(mx.cpu()).asnumpy().item())
        infl_list.append(infl_worker)
        del infl_worker
    infl_list = np.array(infl_list).T.tolist()
    return infl_list


def test(args):
    if args.gpu == -1:
        ctx = mx.cpu()
    else:
        ctx = mx.gpu(args.gpu)

    input_str = ' '.join(sys.argv)
    print(input_str)
    mx.random.seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    global_model_path = args.global_model_path
    num_workers = args.nworkers
    malicious_index = list(range(args.nbyz))
    target = 2

    final_model_file = global_model_path + "epoch" + str(args.epochs - args.ckpt_interval)
    net = load_model(ctx, final_model_file)
    test_data = get_test_data(args)
    if args.byz_type == 'edge':
        path = "data/southwest/"
        with open(path + 'southwest_images_new_test.pkl', 'rb') as train_f:
            saved_southwest_dataset_test = pickle.load(train_f)
        saved_southwest_dataset_test = saved_southwest_dataset_test / 255
        saved_southwest_dataset_test = np.array([saved.T for saved in saved_southwest_dataset_test])
        test_edge_data = nd.array(saved_southwest_dataset_test).as_in_context(ctx)
    try:
        worker_list = np.loadtxt(global_model_path + "chosen_client.txt")
    except:
        worker_list = [list(range(num_workers)) for _ in range(args.epochs)]
    
    true_labels = [0] * num_workers
    for i in malicious_index:
        true_labels[i] = 1

    print("Test result: ")
    detect_list = [0, 1, 2, 3, 4]
    n_samples = len(detect_list)
    if args.byz_type == 'scale':
        backdoor_acc, x_ta, y_label, y_pred, _ = evaluate_accuracy(test_data, net, trigger=True,
                                                                    target=target, context=ctx,
                                                                    args=args)
    elif args.byz_type == 'edge':
        backdoor_acc, x_ta, y_label, y_pred, _ = evaluate_edge_backdoor(test_edge_data, net, context=ctx)
    print(x_ta[0].shape)
    print(len(x_ta))
    print("First sample: pred: ", y_pred[0].asnumpy().item(), "label: ", y_label[0].asnumpy().item())
    test_accuracy, x_tr, y_trlabel, y_trpred, _ = evaluate_accuracy(test_data, net, trigger=False,
                                                                    target=y_pred[0].asnumpy().item(),
                                                                    context=ctx)
    with ctx:
        x_tr = [nd.random.uniform(0, 1, shape=x_ta[0].shape) for _ in range(5)]
        y_trpred = nd.array([y_pred[0].asnumpy().item()] * 5)
        y_trlabel = nd.array([y_pred[0].asnumpy().item()] * 5)
    for i in detect_list:
        print("Second sample: pred: ", y_trpred[i].asnumpy().item(), "label: ", y_trlabel[i].asnumpy().item())
    print('test result: acc-top1=%.4f, backdoor_acc=%.4f' % (test_accuracy, backdoor_acc))
    
    print("Load test gradient: ")
    test_grad = save_test_grad(x_ta[0], y_label[0], args.epochs, args.ckpt_interval, global_model_path, ctx)
    test_grad_trs = []
    for i in detect_list:
        test_grad_trs.append(save_test_grad(x_tr[i], y_trlabel[i], args.epochs, args.ckpt_interval, global_model_path, ctx))
    
    print("Calculate influence: ")
    infl_list_prefixsum = np.array(influence_prefixsum(test_grad, worker_list, args, ctx))

    for _, test_grad_tr in enumerate(test_grad_trs):
        infl_list = np.array(influence_prefixsum(test_grad_tr, worker_list, args, ctx))
        infl_list_prefixsum = np.array(infl_list_prefixsum)
        infl_list = np.array(infl_list)
        infl_all = np.vstack([infl_list_prefixsum[-1], infl_list[-1]])
        
        _, pred_labels, label_malicious, score_list = get_clustered_infl(infl_all)
        print(score_list)
        print(pred_labels)
        detect_acc, detect_fpr, detect_fnr = evaluation(true_labels, pred_labels)
        print("detect_acc: %.4f, detect_fpr: %.4f, detect_fnr: %.4f. " % (detect_acc, detect_fpr, detect_fnr), label_malicious)
    print(input_str)


if __name__ == "__main__":
    args = get_args()
    test(args)