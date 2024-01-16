import random

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def get_dataset(data_size=1000):
    col_names = _col_names()
    df = pd.read_csv("data/kddcup.data_10_percent_corrected", header=None, names=col_names)
    df = df[0:10000]
    text_l = ['protocol_type', 'service', 'flag', 'land', 'logged_in', 'is_host_login', 'is_guest_login']

    for name in text_l:
        _encode_text_dummy(df, name)

    labels = df['label'].copy()
    labels[labels != 'normal.'] = 0
    labels[labels == 'normal.'] = 1

    df['label'] = labels

    df_train = df.sample(frac=0.5, random_state=42)
    df_test = df.loc[~df.index.isin(df_train.index)]

    x_train_df, y_train_df = _to_xy(df_train, target='label')
    x_test_df, y_test_df = _to_xy(df_test, target='label')

    all_cols = _get_all_cols(x_train_df)
    x_train_cols = _get_cols(x_train_df)
    x_test_cols = _get_cols(x_test_df)

    x_train, y_train = x_train_df.to_numpy().astype(np.float32), y_train_df.to_numpy().astype(np.float32)
    y_train = y_train.flatten().astype(int)
    x_test, y_test = x_test_df.to_numpy().astype(np.float32), y_test_df.to_numpy().astype(np.float32)
    y_test = y_test.flatten().astype(int)

    # 去除训练数据中的异常节点
    x_train = x_train[y_train == 1]
    y_train = y_train[y_train == 1]

    # 获取指定大小的训练数据
    sample_list = list(range(x_train.shape[0]))
    sample_list = random.sample(sample_list, data_size)
    x_train = x_train[sample_list, :]
    y_train = y_train[0:data_size]

    # 获取指定大小的测试数据
    x_test_temp1 = x_test[y_test == 1]
    y_test_temp1 = y_test[y_test == 1]
    x_test_temp2 = x_test[y_test == 0]
    y_test_temp2 = y_test[y_test == 0]

    x_test_temp1, y_test_temp1 = select_test_data(x_test_temp1, y_test_temp1, value=1, num=int(data_size * 0.9))
    x_test_temp2, y_test_temp2 = select_test_data(x_test_temp2, y_test_temp2, value=0, num=int(data_size * 0.1))
    x_test = np.append(x_test_temp1, x_test_temp2, axis=0)
    y_test = np.append(y_test_temp1, y_test_temp2, axis=0)

    # 数据归一化
    for i in range(x_train.shape[1]):
        x_train[:, i] = normalization(x_train[:, i])
        x_test[:, i] = normalization(x_test[:, i])

    scaler = MinMaxScaler()
    scaler.fit(x_train)
    scaler.transform(x_train)

    scaler.transform(x_test)

    dataset = {'x_train': x_train.astype(np.float32), 'y_train': y_train.astype(np.float32),
               'x_test': x_test.astype(np.float32), 'y_test': y_test.astype(np.float32)}

    return df, dataset, x_train_cols, x_test_cols, list(x_train_df.axes[0]), all_cols


def select_test_data(x, y, value=0, num=1000):
    res_x = []
    res_y = []
    count = 0
    for i in range(x.shape[0]):
        if y[i] == value:
            res_x.append(x[i, :].tolist())
            res_y.append(y[i])
            count += 1
            if count == num:
                break
    return np.array(res_x), np.array(res_y)


def normalization(data):
    data = data.tolist()
    min_data, max_data = min(data), max(data)
    if max_data == min_data:
        if max_data != 0:
            for i in range(len(data)):
                data[i] = 1
    else:
        t = max_data - min_data
        for i in range(len(data)):
            data[i] = (data[i] - min_data) / t
    # if not (0 <= max_data <= 1 and 0 <= min_data <= 1):
    #     if max_data == min_data:
    #         if not 0 <= max_data <= 1:
    #             for i in range(len(data)):
    #                 data[i] = 1
    #     else:
    #         t = max_data - min_data
    #         for i in range(len(data)):
    #             data[i] = (data[i] - min_data) / t
    data = np.array(data)
    return data


def _col_names():
    """Column names of the dataframe"""
    return ["duration", "protocol_type", "service", "flag", "src_bytes",
            "dst_bytes", "land", "wrong_fragment", "urgent", "hot", "num_failed_logins",
            "logged_in", "num_compromised", "root_shell", "su_attempted", "num_root",
            "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
            "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
            "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
            "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
            "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
            "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
            "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label"]


def _encode_text_dummy(df, name):
    """Encode text values to dummy variables(i.e. [1,0,0],[0,1,0],[0,0,1]
    for red,green,blue)
    """
    dummies = pd.get_dummies(df.loc[:, name])
    for x in dummies.columns:
        dummy_name = "{}-{}".format(name, x)
        df.loc[:, dummy_name] = dummies[x]
    df.drop(name, axis=1, inplace=True)


def _to_xy(df, target):
    """Converts a Pandas dataframe to the x,y inputs that TensorFlow needs"""
    result = []
    for x in df.columns:
        if x != target:
            result.append(x)
    dummies = df[target]
    df_copy = df.copy().drop(columns=target)
    return df_copy, dummies


def get_train(dataset):
    """Get training dataset for KDD 10 percent"""
    return _get_adapted_dataset("train", dataset)


def get_test(dataset):
    """Get testing dataset for KDD 10 percent"""
    return _get_adapted_dataset("test", dataset)


def _get_adapted_dataset(split, dataset):
    """
    Gets the adapted dataset for the experiments

    Args :
            split (str): train or test
    Returns :
            (tuple): <training, testing> images and labels
    """
    # dataset = get_dataset()
    key_img = 'x_' + split
    key_lbl = 'y_' + split

    # if split != 'train':
    # dataset[key_img], dataset[key_lbl] = _adapt(dataset[key_img],
    #                                             dataset[key_lbl])

    return dataset[key_img], dataset[key_lbl]


def _adapt(x, y, rho=0.2):
    """Adapt the ratio of normal/anomalous data"""

    # Normal data: label =0, anomalous data: label =1

    rng = np.random.RandomState(42)  # seed shuffling

    inliersx = x[y == 0]
    inliersy = y[y == 0]
    outliersx = x[y == 1]
    outliersy = y[y == 1]

    size_outliers = outliersx.shape[0]
    inds = rng.permutation(size_outliers)
    outliersx, outliersy = outliersx[inds], outliersy[inds]

    size_test = inliersx.shape[0]
    out_size_test = int(size_test * rho / (1 - rho))

    outestx = outliersx[:out_size_test]
    outesty = outliersy[:out_size_test]

    testx = np.concatenate((inliersx, outestx), axis=0)
    testy = np.concatenate((inliersy, outesty), axis=0)

    size_test = testx.shape[0]
    inds = rng.permutation(size_test)
    testx, testy = testx[inds], testy[inds]

    return testx, testy


def _get_cols(dataset):
    result, temp = [], []
    for index, row in dataset.iterrows():
        for column in dataset.columns:
            if row[column] != 0:
                temp.append(column)
        result.append(temp[:])
        temp.clear()
    return result


def _get_all_cols(dataset):
    result = []
    for column in dataset.columns:
        result.append(column)
    return result


if __name__ == "__main__":
    get_dataset(data_size=1000)
