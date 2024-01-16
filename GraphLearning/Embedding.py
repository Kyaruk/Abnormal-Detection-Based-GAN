import numpy as np


def convolution_cal(walks, datas):
    result = []
    for walk in walks:
        temp_result = {}
        layer = walks.index(walk)
        data = datas[layer]
        for one_walk in walk:
            if len(one_walk) == 5:
                index_x1, index_x2, index_x3, index_x4, index_x5 = int(one_walk[0]), int(one_walk[1]), int(
                    one_walk[2]), int(one_walk[3]), int(one_walk[4])
                x1, x2, x3, x4, x5 = data[index_x1], data[index_x2], data[index_x3], data[index_x4], data[index_x5]
                embedding_res = get_embedding(x1, x2, x3, x4, x5)
                if index_x3 in temp_result:
                    temp_result[index_x3] = list((embedding_res + np.array(temp_result.get(index_x3))) / 2)
                else:
                    temp_result[index_x3] = embedding_res.tolist()
        temp_result = convolution_cal_alone(temp_result, data)
        result.append(temp_result)
    return result


def convolution_cal_alone(temp_result, data):
    size = data.shape[0]
    for i in range(size):
        if not i in temp_result:
            temp_result[i] = list(data[i])
    return temp_result


def get_embedding(x1, x2, x3, x4, x5):
    node1 = np.array(x1)
    node2 = np.array(x2)
    node3 = np.array(x3)
    node4 = np.array(x4)
    node5 = np.array(x5)
    return (node1 + node2 + node4 + node5) * 0.125 + node3 * 0.5


def test_convolution_cal(walks, datas, tests):
    new_walks = []
    new_data = []
    for walk in walks:
        layer = walks.index(walk)
        data = datas[layer]
        test_data = tests[layer]
        new_data.append(np.append(data, test_data, axis=0))
        test_index = len(data)
        new_walk = []
        for one_walk in walk:
            if len(one_walk) == 5 and int(one_walk[2]) >= test_index:
                new_walk.append(one_walk)
        new_walks.append(new_walk)
    temps = convolution_cal(new_walks, new_data)
    res = []
    for temp in temps:
        temp_res = {}
        index = temps.index(temp)
        for key, value in temp.items():
            size = datas[index].shape[0]
            if key >= size:
                temp_res[key] = value
        res.append(temp_res)
    return res
