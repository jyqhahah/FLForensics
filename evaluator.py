import mxnet as mx
from mxnet import nd
from utils import add_trigger

def evaluate_accuracy(data_iterator, net, trigger=False, target=None, context=mx.cpu(), args=None):
    acc = mx.metric.Accuracy()
    flag = False
    err_label = -1
    x_ta, y_label, y_pred = [], [], []
    cnt = 0
    for i, (data, label) in enumerate(data_iterator):
        data = data.as_in_context(context)
        label = label.as_in_context(context)
        remaining_idx = list(range(data.shape[0]))
        if trigger:
            for example_id in range(data.shape[0]):
                data[example_id] = add_trigger(data[example_id])
            for example_id in range(data.shape[0]):
                if label[example_id] != target:
                    label[example_id] = target
                else:
                    remaining_idx.remove(example_id)
        output = net(data)
        predictions = nd.argmax(output, axis=1)

        data = data[remaining_idx]
        predictions = predictions[remaining_idx]
        label = label[remaining_idx]

        acc.update(preds=predictions, labels=label)
        err_ind = -1
        if i == 0:
            x_ta = [data[err_ind].as_in_context(context).reshape(1, 3, 32, 32)]
            y_label = [label[err_ind].as_in_context(context)]
            y_pred = [predictions[err_ind]]
        if not flag:
            for k in range(len(predictions)):
                pr, la = predictions[k], label[k]
                if pr.shape != la.shape:
                    pr = nd.argmax(pr, axis=-1)
                pr = pr.asnumpy().astype('int32')
                la = la.asnumpy().astype('int32')
                err_ind += 1
                if trigger:
                    if pr == la:
                        cnt += 1
                        if cnt == 1:
                            x_ta = [data[err_ind].as_in_context(context).reshape(1, 3, 32, 32)]
                            y_label = [label[err_ind].as_in_context(context)]
                            y_pred = [predictions[err_ind]]
                        else:
                            x_ta.append(data[err_ind].as_in_context(context).reshape(1, 3, 32, 32))
                            y_label.append(label[err_ind].as_in_context(context))
                            y_pred.append(predictions[err_ind])
                            err_label = err_ind
                        if cnt == 100:
                            flag = True
                            break
                else:
                    if target is not None and la == target and la != pr:
                        cnt += 1
                        if cnt == 1:
                            x_ta = [data[err_ind].as_in_context(context).reshape(1, 3, 32, 32)]
                            y_label = [label[err_ind].as_in_context(context)]
                            y_pred = [predictions[err_ind]]
                        else:
                            x_ta.append(data[err_ind].as_in_context(context).reshape(1, 3, 32, 32))
                            y_label.append(label[err_ind].as_in_context(context))
                            y_pred.append(predictions[err_ind])
                            err_label = err_ind
                        if cnt == 100:
                            flag = True
                            break
    return acc.get()[1], x_ta, y_label, y_pred, err_label


def evaluate_edge_backdoor(data, net, context=mx.cpu()):
    acc = mx.metric.Accuracy()
    output = net(data)
    label = 9 * nd.ones(len(data)).as_in_context(context)
    predictions = nd.argmax(output, axis=1)
    err_ind = -1
    err_indices = []
    x_ta = [data[err_ind].as_in_context(context).reshape(1, 3, 32, 32)]
    y_label = [label[err_ind].as_in_context(context)]
    y_pred = [predictions[err_ind]]
    for pr, la in zip(predictions, label):
        if pr.shape != la.shape:
            pr = nd.argmax(pr, axis=-1)
        pr = pr.asnumpy().astype('int32')
        la = la.asnumpy().astype('int32')
        err_ind += 1
        if pr == la:
            err_indices.append(err_ind)
    x_ta = [data[ind].as_in_context(context).reshape(1, 3, 32, 32) for ind in err_indices]
    y_label = [label[ind].as_in_context(context) for ind in err_indices]
    y_pred = [predictions[ind] for ind in err_indices]
    acc.update(preds=predictions, labels=label)
    return acc.get()[1], x_ta, y_label, y_pred, err_indices