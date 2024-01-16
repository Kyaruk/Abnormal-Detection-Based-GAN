# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import random

import data.kdd as data
import data.Apriori as Apriori
import data.PreProcess as PreProcess
import data.MultPlayer as MultPlayer
import GraphLearning.Embedding as Embedding
import GraphLearning.MyGAN as MyGAN
import GetData
import networkx as nx
import matplotlib.pyplot as plt
from node2vec import Node2Vec
import numpy as np
import sklearn.metrics as metrics
from numba import jit
import time


def file_read(file_path):
    save_list = []
    with open(file_path, "r", encoding='utf-8') as file:
        for line in file:
            line = line.strip('\n')
            save_list.append(line)
        file.close()

    return save_list


def file_write(file_path, write_datas):
    for write_data in write_datas:
        with open(file_path + "_" + str(write_datas.index(write_data)), "w", encoding='utf-8') as file:
            for test in write_data:
                file.writelines(str(test) + "\n")
        file.close()


def rules_to_list(rules):
    result = []
    for rule in rules:
        temp1 = []
        for temp in rule[0]:
            temp1.append(temp)
        temp2 = []
        for temp in rule[1]:
            temp2.append(temp)
        res = rule[2]
        result.append((temp1, temp2, res))
    return result


def get_layers_conf(rule_list, mult_attrs):
    result = []
    for i in range(len(mult_attrs) - 1):
        list1 = mult_attrs[i]
        list2 = mult_attrs[i + 1]
        temp_result = (0, 0)
        set1 = set(list1)
        set2 = set(list2)
        for rule in rule_list:
            rule1 = rule[0]
            rule2 = rule[1]
            if set1 > set(rule1) and set2 > set(rule2):
                lens = len(rule1) + len(rule2)
                if lens > temp_result[0]:
                    temp_result = (lens, rule[2])
        result.append(temp_result)
    return result


def build_mult_layer():
    # 数据集获取
    df, dataset, x_train_cols, x_test_cols, x_train_index, all_cols = data.get_dataset(data_size=1010)
    # print(len(x_train_cols[0]))
    trainx, trainy = data.get_train(dataset)
    # trainx_copy = trainx.copy()
    testx, testy = data.get_test(dataset)

    # 数据集预处理
    temp_data, temp_index, temp_col = PreProcess.attr2num(x_train_cols)
    temp_col = sorted(temp_col.items(), key=lambda s: s[0])
    temp_data = PreProcess.del_common_attr(temp_data, temp_index)
    temp_count, temp_count_list = PreProcess.get_cols_count(temp_data)

    # 数据集关联分析
    # (6, 0.9913499344692005) (2, 0.36710526315789477) (4, 0.6428571428571429)
    rules, attr_list = Apriori.my_apriori(minSupport=0.7, temp_data=temp_data)
    rules_list = rules_to_list(rules)
    mult_attrs = Apriori.get_multlayer_attr(attr_list=attr_list, max_length=5, temp_col=temp_col)
    layers_conf = get_layers_conf(rule_list=rules_list, mult_attrs=mult_attrs)
    mult_col_index = []
    for mult_attr in mult_attrs:
        temp_mul_col_index = []
        for temp_attr in mult_attr:
            for temp_temp_col in temp_col:
                if temp_temp_col[1] == temp_attr:
                    temp_mul_col_index.append(all_cols.index(temp_temp_col[0]))
                    break
        mult_col_index.append(temp_mul_col_index)

    # 获得数据集对应的属性
    true_data = []
    test_data = []
    for temp_temp_col in mult_col_index:
        temp_true_data = trainx[:, temp_temp_col]
        temp_test_data = testx[:, temp_temp_col]
        true_data.append(temp_true_data)
        test_data.append(temp_test_data)

    # 多层网络构建
    sigma = 0.001
    weight = 0.99999
    mult_layer_matrix, sigma_opted, weight_opted, edges = MultPlayer.build_mult_layer(datas=true_data, sigma=sigma,
                                                                                      weight=weight)
    return mult_layer_matrix, sigma_opted, weight_opted, edges, true_data, test_data, testy


def build_test_mult_layer(tests, mult_layer_matrix, true_data, sigma, weight):
    test_mult_layer_matrix, sigma_opted, weight_opted, test_edges = MultPlayer.build_test_mult_layer(tests=tests,
                                                                                                     mult_layer_matrix=mult_layer_matrix,
                                                                                                     datas=true_data,
                                                                                                     sigma=sigma,
                                                                                                     weight=weight)
    return test_mult_layer_matrix, sigma_opted, weight_opted, test_edges


def get_walks(nodes, edge):
    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edge)
    node2vec = my_node2vec(G)
    return G, node2vec


def get_train_embedding(true_data, edges, layer1_nodes):
    nodes = []
    for i in range(true_data[0].shape[0]):
        nodes.append(i)
    Gs = []
    walks = []
    for edge in edges:
        if edges.index(edge) == 1:
            G, node2vec = get_walks(nodes=layer1_nodes, edge=edge)
            Gs.append(G)
            walks.append(node2vec.walks)
        else:
            G, node2vec = get_walks(nodes=nodes, edge=edge)
            Gs.append(G)
            walks.append(node2vec.walks)
        # TODO
        nx.draw(G, pos=nx.random_layout(G), node_size=10, node_shape='o', width=0.3, style='solid', font_size=8)
        # plt.show()

    embeddings = Embedding.convolution_cal(walks=walks, datas=true_data)

    # embeddings : [{int : [], int : []},{}]
    return embeddings, Gs


def get_test_embedding():
    return 0


# @jit()
def my_node2vec(graph):
    return Node2Vec(graph, dimensions=64, walk_length=5, num_walks=5000, workers=1)


def get_sorted_list(D_res, reverse=False):
    result = []
    key = list(D_res.keys())
    key.sort(reverse=reverse)
    sorted_list = [(temp_key, D_res[temp_key]) for temp_key in key]
    for temp in sorted_list:
        result.append(temp[1])
    return result


def result_analysis_total(G_res_total, D_res_total, labels):
    init_value = 1
    res = []
    G_test_labels = [init_value for i in range(len(labels))]
    D_test_labels = [init_value for i in range(len(labels))]
    size = len(labels)
    for G_res, D_res in zip(G_res_total, D_res_total):
        temp_res = []
        for key, value in G_res.items():
            if value < 35:
                G_test_labels[key - size] = 0
        for key, value in D_res.items():
            if value < 0.5:
                D_test_labels[key - size] = 0
                temp_res.append(key - size)
        res.append(temp_res)
    return G_test_labels, D_test_labels


def result_analysis(G_res, D_res, size, labels, index):
    test_size = labels.shape[0]
    list_roc = np.arange(0, 1, 0.001)
    roc_x_fpr_G = []
    roc_y_tpr_G = []
    roc_x_fpr_D = []
    roc_y_tpr_D = []
    G_max_recall = 0
    G_max_acc = 0
    G_max_pre = 0
    G_max_f1 = 0
    D_max_recall = 0
    D_max_acc = 0
    D_max_pre = 0
    D_max_f1 = 0
    G_max_TP, G_max_FP, G_max_TN, G_max_FN = 0, 0, 0, 0
    D_max_TP, D_max_FP, D_max_TN, D_max_FN = 0, 0, 0, 0
    sorted_D_res = get_sorted_list(D_res)
    random_list = list(range(0, len(sorted_D_res)))
    random.shuffle(random_list)
    my_ROC(labels, np.array(sorted_D_res), index=index)
    # plot_ROC(labels, np.array(sorted_D_res))
    for temp_value_roc in list_roc:
        value_roc = round(temp_value_roc, 3)
        min_g, max_g = 0, 0
        sum_g = 0
        G_TP, G_FP, G_TN, G_FN = 0, 0, 0, 0
        D_TP, D_FP, D_TN, D_FN = 0, 0, 0, 0
        y = []
        y2 = []
        for G_key, G_value in G_res.items():
            min_g = G_value
            max_g = G_value
            break
        for G_key, G_value in G_res.items():
            min_g = min(min_g, G_value)
            max_g = max(max_g, G_value)
            sum_g += G_value
            y.append(G_value)
            temp_G_key = G_key - size - 1
            if G_value < 35:
                if 0 <= temp_G_key < test_size and labels[temp_G_key] == 1:
                    G_TP += 1
                else:
                    G_FP += 1
            else:
                if test_size > temp_G_key >= 0 == labels[temp_G_key]:
                    G_TN += 1
                else:
                    G_FN += 1

        print("value_roc = " + str(value_roc))
        print("min_g = " + str(min_g) + ", max_g = " + str(max_g) + ", mean_g = " + str(sum_g / test_size))

        for D_key, D_value in D_res.items():
            temp_D_key = D_key - size - 1
            y2.append(D_value)
            if D_value > value_roc:
                if 0 <= temp_D_key < test_size and labels[temp_D_key] == 1:
                    D_TP += 1
                else:
                    D_FP += 1
            else:
                if test_size > temp_D_key >= 0 == labels[temp_D_key]:
                    D_TN += 1
                else:
                    D_FN += 1

        G_recall = get_recall(TP=G_TP, FN=G_FN)
        G_acc = get_acc(TP=G_TP, TN=G_TN, FP=G_FP, FN=G_FN)
        G_pre = get_pre(TP=G_TP, FP=G_FP)
        G_f1 = get_f1(pre=G_pre, recall=G_recall)

        D_recall = get_recall(TP=D_TP, FN=D_FN)
        D_acc = get_acc(TP=D_TP, TN=D_TN, FP=D_FP, FN=D_FN)
        D_pre = get_pre(TP=D_TP, FP=D_FP)
        D_f1 = get_f1(pre=D_pre, recall=D_recall)

        print(
            "G_recall = " + str(G_recall) + ", G_acc = " + str(G_acc) + ", G_pre = " + str(G_pre) + ", G_f1 = " + str(
                G_f1))
        print(
            "D_recall = " + str(D_recall) + ", D_acc = " + str(D_acc) + ", D_pre = " + str(D_pre) + ", D_f1 = " + str(
                D_f1))
        print()

        roc_x_fpr_G.append(get_fpr(FP=G_FP, TN=G_TN))
        roc_y_tpr_G.append(get_tpr(TP=G_TP, FN=G_FN))
        roc_x_fpr_D.append(get_fpr(FP=D_FP, TN=D_TN))
        roc_y_tpr_D.append(get_tpr(TP=D_TP, FN=D_FN))

        if G_max_recall < G_recall:
            G_max_recall = G_recall
        if G_max_acc < G_acc:
            G_max_acc = G_acc
        if G_max_pre < G_pre:
            G_max_pre = G_pre
        if G_max_f1 < G_f1:
            G_max_f1 = G_f1
        if D_max_recall < D_recall:
            D_max_recall = D_recall
        if D_max_acc < D_acc:
            D_max_acc = D_acc
        if D_max_pre < D_pre:
            D_max_pre = D_pre
        if D_max_f1 < D_f1:
            D_max_f1 = D_f1
            D_max_TP = D_TP
            D_max_FP = D_FP
            D_max_TN = D_TN
            D_max_FN = D_FN

        if G_max_TP <= G_TP:
            G_max_TP = G_TP
        if G_max_FP <= G_FP:
            G_max_FP = G_FP
        if G_max_TN <= G_TN:
            G_max_TN = G_TN
        if G_max_FN <= G_FN:
            G_max_FN = G_FN
        # if D_max_TP <= D_TP:
        #     D_max_TP = D_TP
        # if D_max_FP <= D_FP:
        #     D_max_FP = D_FP
        # if D_max_TN <= D_TN:
        #     D_max_TN = D_TN
        # if D_max_FN <= D_FN:
        #     D_max_FN = D_FN

    # plt.scatter(roc_x_fpr_G, roc_y_tpr_G, color="orange", marker="o", label="G_roc")
    # plt.plot(roc_x_fpr_G, roc_y_tpr_G)
    # plt.title("ROC_G", fontsize=24)
    # plt.grid()
    # plt.show()
    # if index == 0:
    #     plt.subplot(221)
    # elif index == 1:
    #     plt.subplot(222)
    # elif index == 2:
    #     plt.subplot(223)
    # else:
    #     plt.subplot(224)
    # print("roc_x_fpr_D = " + str(roc_x_fpr_D))
    # print("roc_y_tpr_D = " + str(roc_y_tpr_G))
    # plt.scatter(roc_x_fpr_D, roc_y_tpr_D, color="orange", marker="o", label="D_roc")
    # plt.grid()
    print("G_max_recall = " + str(G_max_recall))
    print("G_max_acc = " + str(G_max_acc))
    print("G_max_pre = " + str(G_max_pre))
    print("G_max_f1 = " + str(G_max_f1))
    print("D_max_recall = " + str(D_max_recall))
    print("D_max_acc = " + str(D_max_acc))
    print("D_max_pre = " + str(D_max_pre))
    print("D_max_f1 = " + str(D_max_f1))

    return G_max_recall, G_max_acc, G_max_pre, G_max_f1, D_max_recall, D_max_acc, D_max_pre, D_max_f1, G_max_TP, G_max_FP, G_max_TN, G_max_FN, D_max_TP, D_max_FP, D_max_TN, D_max_FN, sorted_D_res


def my_ROC(y_test, y_score, index=0):
    fpr, tpr, threshold = metrics.roc_curve(y_test, y_score)
    roc_auc = metrics.auc(fpr, tpr)
    plt.figure()
    if index == 0:
        plt.subplot(221)
    elif index == 1:
        plt.subplot(222)
    elif index == 2:
        plt.subplot(223)
    else:
        plt.subplot(224)
    lw = 3
    plt.figure(figsize=(10, 10))
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=22)
    plt.ylabel('True Positive Rate', fontsize=22)
    plt.tick_params(labelsize=22)
    plt.title('ROC曲线 Layer ' + str(index), fontsize=22)
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.legend(loc="lower right")
    # plt.show()


def get_recall(TP, FN):
    if TP + FN == 0:
        return 0
    return TP / (TP + FN)


def get_acc(TP, TN, FP, FN):
    if TP + TN + FP + FN == 0:
        return 0
    return (TP + TN) / (TP + TN + FP + FN)


def get_pre(TP, FP):
    if TP + FP == 0:
        return 0
    return TP / (TP + FP)


def get_f1(pre, recall):
    if pre + recall == 0:
        return 0
    return 2 * pre * recall / (pre + recall)


def get_tpr(TP, FN):
    if TP + FN == 0:
        return 0
    return TP / (TP + FN)


def get_fpr(FP, TN):
    if FP + TN == 0:
        return 0
    return FP / (FP + TN)


def matrix_deal(mult_layer_matrix):
    array2 = mult_layer_matrix[1]
    res = []
    for i in range(array2.shape[0]):
        row = array2[i, :]
        if row[0] == row[1] == row[2] == 0:
            res.append(i)
    temp_index = 0
    num = 3 * len(res)
    for i in range(array2.shape[0]):
        if i not in res:
            res.append(i)
            temp_index = i
            num -= 1
            if num == 0:
                break
    res.sort()
    length = len(res)
    temp = array2[temp_index, res]
    temp = temp.reshape(1, length)
    for i in res:
        if i != temp_index:
            temp = np.append(temp, array2[i, res].reshape(1, length), axis=0)

    return res, temp_index, temp


def test_matrix_deal(mult_layer_matrix, layer1_nodes, true_data_length=1000):
    array2 = mult_layer_matrix[1]
    size = len(layer1_nodes)
    res = []
    for i in range(array2.shape[0]):
        row = array2[i, :]
        if i < size:
            res.append(i)
        else:
            if row[0] == row[1] == row[2] == 0:
                res.append(i)
    temp_index = 0
    num = 3 * len(res)
    for i in range(size, array2.shape[0]):
        if i not in res:
            res.append(i)
            temp_index = i
            num -= 1
            if num == 0:
                break
    res.sort()
    length = len(res)
    temp = array2[temp_index, res]
    temp = temp.reshape(1, length)
    for i in res:
        if i != temp_index:
            temp = np.append(temp, array2[i, res].reshape(1, length), axis=0)
    for i in range(len(layer1_nodes)):
        res[i] = layer1_nodes[i]
    for i in range(len(layer1_nodes), len(res)):
        if res[i] < true_data_length:
            res[i] += true_data_length

    return res, temp_index, temp


def get_layer1_edge(layer1_nodes, layer1_matrix):
    layer1_edge = []
    layer1_size = layer1_matrix.shape[0]
    for i in range(layer1_size):
        for j in range(i + 1, layer1_size):
            if layer1_matrix[i, j] == 1:
                layer1_edge.append((layer1_nodes[i], layer1_nodes[j]))
    return layer1_edge


def predeal_D_res(D_res, index=0):
    result = {}
    temp_value = 0.5
    if index == 1 or index == 0:
        for key, value in D_res.items():
            D_res[key] = 1 - value
    for key, value in D_res.items():
        if value > temp_value:
            result[key] = (value - temp_value) / (1 - temp_value) * (1 - 0.8) + 0.8
        else:
            result[key] = value / (1 - temp_value) * 0.2
    return result


def plot_ROC(labels, preds):
    fpr1, tpr1, threshold1 = metrics.roc_curve(labels, preds)  ###计算真正率和假正率

    roc_auc1 = metrics.auc(fpr1, tpr1)  ###计算auc的值，auc就是曲线包围的面积，越大越好
    plt.figure()
    lw = 3
    plt.figure(figsize=(10, 10))
    plt.plot(fpr1, tpr1, color='darkorange',
             lw=lw, label='AUC = %0.2f' % roc_auc1)  ###假正率为横坐标，真正率为纵坐标做曲线
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('1 - Specificity', fontsize=22)
    plt.ylabel('Sensitivity', fontsize=22)
    plt.tick_params(labelsize=22)
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False
    # plt.title('ROCs for Densenet')
    plt.legend(loc="lower right")
    # plt.show()
    # plt.savefig(savepath)  # 保存文件

def write_to_file(data, file_path):
    with open(file_path, "w", encoding='utf-8') as file:
        file.write(str(data))
        # for test in data:
        #     file.writelines(str(test) + "\n")
    file.close()


if __name__ == "__main__":

    # my_get_data = GetData.GetData(data_size=100)
    # true_data, test_data, labels, attrs = my_get_data.get_data()
    # sigma = 0.001
    # weight = 0.99999
    # mult_layer_matrix, sigma_opted, weight_opted, edges = MultPlayer.build_mult_layer(datas=true_data, sigma=sigma,
    #                                                                                   weight=weight)

    mult_layer_matrix, sigma_opted, weight_opted, edges, true_data, test_data, labels = build_mult_layer()

    layer1_nodes, layer1_temp_index, layer1_matrix = matrix_deal(mult_layer_matrix)

    mult_layer_matrix[1] = layer1_matrix
    edges[1] = get_layer1_edge(layer1_nodes, layer1_matrix)

    embeddings, Gs = get_train_embedding(true_data=true_data, edges=edges, layer1_nodes=layer1_nodes)

    # print("sigma_opted = " + str(sigma_opted))
    # print("weight_opted = " + str(weight_opted))

    my_gans = []
    n_ideas = []

    for embedding in embeddings:
        for key, value in embedding.items():
            n_ideas.append(len(value))
            break

    for embedding in embeddings:
        index = embeddings.index(embedding)
        my_gan = MyGAN.MyGAN(N_IDEAS=n_ideas[index], BATCH_SIZE=1000, data=embedding)
        G, D = my_gan.learning()
        my_gans.append(my_gan)

    test_mult_layer_matrix, test_sigma_opted, test_weight_opted, test_edges = build_test_mult_layer(tests=test_data,
                                                                                                    mult_layer_matrix=mult_layer_matrix,
                                                                                                    true_data=true_data,
                                                                                                    sigma=0.01,
                                                                                                    weight=0.99999)
    layer1_test_nodes, layer1_test_temp_index, layer1_test_matrix = test_matrix_deal(test_mult_layer_matrix,
                                                                                     layer1_nodes,
                                                                                     true_data[0].shape[0])
    test_mult_layer_matrix[1] = layer1_test_matrix
    test_edges[1] = get_layer1_edge(layer1_test_nodes, layer1_test_matrix)

    test_walks = []
    test_Gs = []
    test_nodes = list(range(true_data[0].shape[0] + test_data[0].shape[0]))
    for test_edge in test_edges:
        if test_edges.index(test_edge) == 1:
            G, node2vec = get_walks(nodes=layer1_test_nodes, edge=test_edge)
            test_Gs.append(G)
            test_walks.append(node2vec.walks)
        else:
            G, node2vec = get_walks(nodes=test_nodes, edge=test_edge)
            test_Gs.append(G)
            test_walks.append(node2vec.walks)
            # print(node2vec)
        # TODO: del print()
        # print("debug" + str(test_edges.index(test_edge)))
        nx.draw(G, pos=nx.random_layout(G), node_size=10, node_shape='o', width=0.3, style='solid', font_size=8)
        # plt.show()

    test_embeddings = Embedding.test_convolution_cal(walks=test_walks, datas=true_data, tests=test_data)

    G_max_recalls = []
    G_max_accs = []
    G_max_pres = []
    G_max_f1s = []
    D_max_recalls = []
    D_max_accs = []
    D_max_pres = []
    D_max_f1s = []

    G_max_TPs = []
    G_max_FPs = []
    G_max_TNs = []
    G_max_FNs = []
    D_max_TPs = []
    D_max_FPs = []
    D_max_TNs = []
    D_max_FNs = []
    G_res_total = []
    D_res_total = []

    for test_embedding, test_gan in zip(test_embeddings, my_gans):
        index_file = test_embeddings.index(test_embedding)
        D_res = test_gan.detect_D(test_embedding)
        G_res = test_gan.detect_G(test_embedding)
        D_res = predeal_D_res(D_res, index=test_embeddings.index(test_embedding))
        print("Layer " + str(test_embeddings.index(test_embedding)) + ":")
        # print("D_res = " + str(D_res))
        # print("G_res = " + str(G_res))
        G_res_total.append(G_res)
        D_res_total.append(D_res)
        G_max_recall, G_max_acc, G_max_pre, G_max_f1, D_max_recall, D_max_acc, D_max_pre, D_max_f1, G_max_TP, G_max_FP, G_max_TN, G_max_FN, D_max_TP, D_max_FP, D_max_TN, D_max_FN, D_labels = result_analysis(
            G_res=G_res, D_res=D_res, size=true_data[0].shape[0], labels=labels,
            index=my_gans.index(test_gan))
        write_to_file(D_labels, "Res" + str(index_file))
        G_max_recalls.append(G_max_recall)
        G_max_accs.append(G_max_acc)
        G_max_pres.append(G_max_pre)
        G_max_f1s.append(G_max_f1)
        D_max_recalls.append(D_max_recall)
        D_max_accs.append(D_max_acc)
        D_max_pres.append(D_max_pre)
        D_max_f1s.append(D_max_f1)

        G_max_TPs.append(G_max_TP)
        G_max_FPs.append(G_max_FP)
        G_max_TNs.append(G_max_TN)
        G_max_FNs.append(G_max_FN)
        D_max_TPs.append(D_max_TP)
        D_max_FPs.append(D_max_FP)
        D_max_TNs.append(D_max_TN)
        D_max_FNs.append(D_max_FN)

    plt.show()

    G_test_labels, D_test_labels = result_analysis_total(G_res_total=G_res_total, D_res_total=D_res_total,
                                                         labels=labels)
    my_ROC(labels, np.array(G_test_labels), index=index)
    # plot_ROC(labels, np.array(G_test_labels))
    my_ROC(labels, np.array(D_test_labels), index=index)
    # plot_ROC(labels, np.array(D_test_labels))
    plt.show()

    print("G_max_recall = " + str(G_max_recalls))
    print("G_max_acc = " + str(G_max_accs))
    print("G_max_pre = " + str(G_max_pres))
    print("G_max_f1 = " + str(G_max_f1s))
    print("D_max_recall = " + str(D_max_recalls))
    print("D_max_acc = " + str(D_max_accs))
    print("D_max_pre = " + str(D_max_pres))
    print("D_max_f1 = " + str(D_max_f1s))

    print("G_max_TP = " + str(G_max_TPs))
    print("G_max_FP = " + str(G_max_FPs))
    print("G_max_TN = " + str(G_max_TNs))
    print("G_max_FN = " + str(G_max_FNs))
    print("D_max_TP = " + str(D_max_TPs))
    print("D_max_FP = " + str(D_max_FPs))
    print("D_max_TN = " + str(D_max_TNs))
    print("D_max_FN = " + str(D_max_FNs))
    print(D_labels)

    # plt.show()
    # print("debug")
    # print(torch.__version__)
    # print(torch.version.cuda)
    # print(torch.backends.cudnn.version())
    # print(torch.cuda.is_available())
    # print(torch.cuda.device_count())
