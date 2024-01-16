# 数据集属性特征数字化表示
def attr2num(data):
    temp_data = []
    temp_col = {}
    temp_index = 0
    for x_train_col in data:
        temp_temp_data = []
        for col in x_train_col:
            if col not in temp_col:
                temp_col[col] = temp_index
                temp_index += 1
            if col in temp_col:
                temp_temp_data.append(temp_col[col])
        temp_data.append(temp_temp_data)
    return temp_data, temp_index, temp_col


# 删除数据集中共有的属性
def del_common_attr(data, temp_index):
    temp_same_col = []
    flag = 0
    for i in range(temp_index):
        flag = 0
        for temp_temp_data in data:
            if i not in temp_temp_data:
                flag = 1
                break
        if flag == 0:
            temp_same_col.append(i)

    temp_data_del = data[:]
    temp_data = []
    for temp_temp_data in temp_data_del:
        temp_temp_data = list(filter(lambda x: x not in temp_same_col, temp_temp_data))
        temp_data.append(sorted(temp_temp_data))

    temp_num = {}
    temp_num_list = []
    for temp_temp_data in temp_data:
        if temp_temp_data not in temp_num_list:
            temp_num_list.append(temp_temp_data)
            temp_num[temp_num_list.index(temp_temp_data)] = 1
        else:
            temp_num[temp_num_list.index(temp_temp_data)] += 1
    return temp_data


# 获得数据集中不同特征值的数量分布
def get_cols_count(data):
    temp_count = {}
    temp_count_list = []
    for temp_temp_data in data:
        if temp_temp_data not in temp_count_list:
            temp_count_list.append(temp_temp_data)
            temp_count[temp_count_list.index(temp_temp_data)] = 1
        else:
            temp_count[temp_count_list.index(temp_temp_data)] += 1
    return temp_count, temp_count_list
