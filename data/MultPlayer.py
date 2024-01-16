import math

import numpy as np
from numba import jit


def build_mult_layer(datas, sigma=100, weight=0.9999):
    result = []
    counts = []
    edges = []

    sigma_opted = []
    weight_opted = []

    temp_sigma = 0.1
    temp_weight = 0.99993
    print("sigma = " + str(temp_sigma) + ", weight = " + str(temp_weight))
    temp_res, temp_count, edge = build_layer(data=datas[0], sigma=temp_sigma, weight=temp_weight)
    result.append(temp_res)
    counts.append(temp_count)
    edges.append(edge)

    for data in datas[1:]:
        temp_sigma = sigma
        temp_weight = weight
        print("sigma = " + str(temp_sigma) + ", weight = " + str(temp_weight))
        temp_res, temp_count, edge = build_layer(data=data, sigma=temp_sigma, weight=temp_weight)
        result.append(temp_res)
        counts.append(temp_count)
        edges.append(edge)
    return result, sigma_opted, weight_opted, edges


def build_layer(data, sigma=0.1, weight=0.9999):
    num = data.shape[0]
    shape = (num, num)
    matrix = np.zeros(shape)
    edges = []
    count = 0
    for i in range(num):
        for j in range(i + 1, num):
            if weight < get_distance(data[i], data[j], sigma=sigma):
                count += 1
                matrix[i][j] = 1
                matrix[j][i] = 1
                edges.append((i, j))
    print(count)
    return matrix, count, edges


# @jit()
def get_distance(x1, x2, sigma=0.1):
    temp1 = np.array(x1)
    temp2 = np.array(x2)
    dis = np.linalg.norm(temp1 - temp2)
    result = math.exp(-(dis * dis) / (2 * sigma * sigma))
    return result


def build_test_mult_layer(tests, mult_layer_matrix, datas, sigma=0.1, weight=0.9999):
    matrix = mult_layer_matrix
    result = []
    counts = []
    edges = []

    sigma_opted = []
    weight_opted = []

    temp_sigma = 0.1
    temp_weight = 0.99993
    print("sigma = " + str(temp_sigma) + ", weight = " + str(temp_weight))
    temp_res, temp_count, edge = build_test_layer(test=tests[0], mult_matrix=matrix[0], data=datas[0], sigma=temp_sigma,
                                                  weight=temp_weight)
    result.append(temp_res)
    counts.append(temp_count)
    edges.append(edge)

    for test, data, temp_matrix in zip(tests[1:], datas[1:], matrix[1:]):
        temp_sigma = sigma
        temp_weight = weight
        print("sigma = " + str(temp_sigma) + ", weight = " + str(temp_weight))
        temp_res, temp_count, edge = build_test_layer(test=test, mult_matrix=temp_matrix, data=data, sigma=temp_sigma,
                                                      weight=temp_weight)
        result.append(temp_res)
        counts.append(temp_count)
        edges.append(edge)
    return result, sigma_opted, weight_opted, edges


def build_test_layer(test, mult_matrix, data, sigma=0.1, weight=0.9999):
    train_num = len(mult_matrix[0])
    test_num = test.shape[0]
    num = test_num + train_num
    shape = (num, num)
    matrix = np.zeros(shape)
    data = np.append(data, test, axis=0)
    edges = []
    count = 0
    for i in range(num):
        for j in range(i + 1, num):
            if i < train_num and j < train_num:
                matrix[i][j] = mult_matrix[i][j]
                matrix[j][i] = mult_matrix[j][i]
            else:
                if weight < get_distance(data[i], data[j], sigma=sigma):
                    count += 1
                    matrix[i][j] = 1
                    matrix[j][i] = 1
                    edges.append((i, j))
    print(count)
    return matrix, count, edges
