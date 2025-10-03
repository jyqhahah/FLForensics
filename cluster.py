import numpy as np
from sklearn.metrics import confusion_matrix
import hdbscan

def evaluation(y_true, y_prediction):
    cnf_matrix = confusion_matrix(y_true, y_prediction)

    FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)
    FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
    TP = np.diag(cnf_matrix)
    TN = cnf_matrix.sum() - (FP + FN + TP)

    FP = FP.astype(float)
    FN = FN.astype(float)
    TP = TP.astype(float)
    TN = TN.astype(float)

    FPR = FP / (FP + TN)
    FNR = FN / (TP + FN)
    ACC = (TP + TN) / (TP + FP + FN + TN)
    return ACC[1], FPR[1], FNR[1]

def check(alpha, infl_all_, tmp_label, label_all):
    if alpha == 1.0:
        return True
    for label in tmp_label:
        tmp_list = [i for i in range(len(label_all)) if label_all[i] == label]
        score1 = np.mean(np.array(infl_all_[0])[tmp_list])
        score2 = np.mean(np.array(infl_all_[1])[tmp_list])
        if score2 / score1 >= alpha and score1 / score2 >= alpha:
            continue
        else:
            return True
    return False

def get_clustered_infl(infl_all_, alpha=1.0):
    s1_max, s1_min = np.max(infl_all_[0]), np.min(infl_all_[0])
    s2_max, s2_min = np.max(infl_all_[1]), np.min(infl_all_[1])
    infl_all = np.array([infl_all_[0] / (s1_max - s1_min), infl_all_[1] / (s2_max - s2_min)])
    min_cluster_size = 7 
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=1, metric="euclidean")
    clusterer.fit(infl_all.T)
    print(clusterer.labels_)
    label_all = clusterer.labels_
    label_set = set(label_all)
    score_list = {}
    tmp_label = []
    for label in label_set:
        tmp_list = [i for i in range(len(label_all)) if label_all[i] == label]
        score1 = np.mean(np.array(infl_all[0])[tmp_list])
        score2 = np.mean(np.array(infl_all[1])[tmp_list])
        score_list[label] = [score1, score2]
        if score1 > 0 and len(tmp_list) < len(label_all) / 2: tmp_label.append(label)
    potential_labels = []
    for i in range(len(label_all)):
        if label_all[i] in tmp_label and label_all[i] != -1:
            potential_labels.append(i)
    infl_pos = infl_all[:, potential_labels]
    mean_pos = np.mean(infl_pos[1]) / np.mean(infl_pos[0])
    print("mean_all: ", mean_pos, "  ", alpha)
    if len(tmp_label) > 1:
        cate1_labels = []
        flag = check(alpha, infl_all_, tmp_label, label_all)
        if not flag:
            label_malicious = []
            pred_labels = [0] * len(label_all)
        else:
            label_malicious = []
            for label in tmp_label:
                s1, s2 = score_list[label][0], score_list[label][1]
                print(label, s2 / s1)
                if s2 / s1 <= mean_pos and label != -1:
                    label_malicious.append(label)
            pred_labels = []
            cate1_labels = []
            for i in label_all:
                if i in label_malicious:
                    pred_labels.append(1)
                    cate1_labels.append(0)
                elif i in tmp_label:
                    pred_labels.append(0)
                    cate1_labels.append(1)
                else:
                    pred_labels.append(0)
                    cate1_labels.append(0)
            for i in range(len(label_all)):
                if label_all[i] == -1 and infl_all[0][i] > 0:
                    if infl_all[1][i] / infl_all[0][i] <= mean_pos:
                        pred_labels[i] = 1
                    else:
                        cate1_labels[i] = 1
    elif len(tmp_label) == 1:
        cate1_labels = []
        flag = check(alpha, infl_all_, tmp_label, label_all)
        if not flag:
            label_malicious = []
            pred_labels = [0] * len(label_all)
        else:
            label_malicious = tmp_label
            pred_labels = [0] * len(label_all)
            for i in range(len(label_all)):
                if label_all[i] == label_malicious[0]:
                    pred_labels[i] = 1
    else:
        label_malicious = []
        cate1_labels = []
        pred_labels = [0] * len(label_all)
    return label_all, pred_labels, label_malicious, score_list

