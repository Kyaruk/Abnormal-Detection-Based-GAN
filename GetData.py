import csv
import re
import numpy as np
import pandas as pd
import random


class GetData:

    def __init__(self, data_size):
        self.data_size = data_size
        self.data = []
        self.true_data = []
        self.test_data = []
        self.labels = []
        self.attrs = []

    def get_data(self):
        # self.get_nsl_kdd()
        label_index = self.get_unsw_nb15()
        # self.get_Tor()
        # self.get_CIC_IDS_2017()

        # for attr in self.attrs:
        #     self.data.append(data_array[:, attr])

        # self.data = np.column_stack((np.arange(0, self.data.shape[0]), self.data))
        # label_index += 1
        # label_index = self.data.shape[1] - 1

        df = pd.DataFrame(self.data)
        df_train = df.sample(frac=0.5, random_state=42)
        df_test = df.loc[~df.index.isin(df_train.index)]

        y_train = df_train[label_index].values
        x_train = df_train.copy().drop(columns=label_index).values
        y_test = df_test[label_index].values
        x_test = df_test.copy().drop(columns=label_index).values

        x_train = x_train[y_train == '1']
        y_train = y_train[y_train == '1']

        sample_list = list(range(x_train.shape[0]))
        sample_list = random.sample(sample_list, self.data_size)
        x_train = x_train[sample_list, :]
        y_train = y_train[0:self.data_size]

        x_test_temp1 = x_test[y_test == '1']
        y_test_temp1 = y_test[y_test == '1']
        x_test_temp2 = x_test[y_test == '0']
        y_test_temp2 = y_test[y_test == '0']

        x_test_temp1, y_test_temp1 = self.select_test_data(x_test_temp1, y_test_temp1, value=1,
                                                           num=int(self.data_size * 0.9))
        x_test_temp2, y_test_temp2 = self.select_test_data(x_test_temp2, y_test_temp2, value=0,
                                                           num=int(self.data_size * 0.1))
        x_test = np.append(x_test_temp1, x_test_temp2, axis=0)
        y_test = np.append(y_test_temp1, y_test_temp2, axis=0)

        array_train = x_train
        array_test = x_test

        for attr in self.attrs:
            self.true_data.append(array_train[:, attr])
            self.test_data.append(array_test[:, attr])

        self.labels = y_test

        return self.true_data, self.test_data, self.labels, self.attrs

    def get_nsl_kdd(self):
        file_name = "C:/Users/14126/Desktop/毕设/数据集/NSL-KDD/KDDTrain+.txt"
        with open(file_name, 'r', encoding='utf-8') as file:
            data = file.readlines()
            # print(data)

        with open('data/NSL-KDD.txt', 'w', encoding="utf-8") as file:
            # writer = csv.writer(file)
            # Use writerows() not writerow()
            # writer.writerows(new_data)
            for w_data in data:
                file.writelines(self.deal_nsl_kdd(w_data))

    def get_unsw_nb15(self):
        file_name = "C:/Users/14126/Desktop/毕设/数据集/UNSW-NB15/training and testing set/UNSW_NB15_training-set.csv"
        datas = []
        with open(file_name, encoding="utf-8") as f:
            f_csv = csv.reader(f)
            headers = next(f_csv)
            for row in f_csv:
                datas.append(self.deal_unsw_nb15(row))
                # print(row)
        # print(datas)
        new_data = self.data_to_str(datas)

        for header in headers:
            print("\"" + header + "\", ", end="")

        with open('data/UNSW-NB15.txt', 'w', encoding="utf-8") as file:
            # writer = csv.writer(file)
            # Use writerows() not writerow()
            # writer.writerows(new_data)
            for w_data in new_data:
                file.writelines(w_data)
        data_array = np.array(datas)

        data_array = np.delete(data_array, 0, 1)
        del headers[-2]
        del headers[0]

        names = ['proto', 'service', 'state']
        df = pd.DataFrame(data_array)
        df.columns = headers
        print(df.loc)
        for name in names:
            dummies = pd.get_dummies(df.loc[:, name])
            dummies_columns = []
            for x in dummies.columns:
                dummy_name = "{}-{}".format(name, x)
                dummies_columns.append(dummy_name)
                # df.insert(0, dummy_name, dummies[x])
            dummies.columns = dummies_columns
            df = pd.concat([df, dummies], axis=1)
            df.drop(name, axis=1, inplace=True)

        self.attrs, label_index = self.get_layer_attr(headers=df.columns.values, name="UNSW-NB15")

        data_array = df.values
        self.labels = data_array[:, label_index]

        for i in range(data_array.shape[1]):
            data_array[:, i] = self.normalization(data_array[:, i])
        self.data = data_array
        return label_index

    def get_Tor(self):
        file_name = "C:/Users/14126/Desktop/毕设/数据集/TorCSV/Scenario-A/SelectedFeatures-10s-TOR-NonTOR.csv"
        datas = []
        with open(file_name, encoding="utf-8") as f:
            f_csv = csv.reader(f)
            headers = next(f_csv)
            for row in f_csv:
                datas.append(self.deal_unsw_nb15(row))
                # print(row)
        # print(datas)
        headers = self.del_headers_spqce(headers=headers)
        self.attrs, label_index = self.get_layer_attr(headers=headers, name="Tor")
        new_data = self.data_to_str(datas)
        with open('data/Tor.txt', 'w', encoding="utf-8") as file:
            # writer = csv.writer(file)
            # Use writerows() not writerow()
            # writer.writerows(new_data)
            for w_data in new_data:
                file.writelines(w_data)
        data_array = np.array(datas)

        self.labels = data_array[:, label_index]
        self.labels[self.labels != "nonTOR"] = 0
        self.labels[self.labels == "nonTOR"] = 1

        del_index = [0, 2, data_array.shape[1] - 1]
        for i in range(data_array.shape[1]):
            if i not in del_index:
                data_array[:, i] = self.normalization(data_array[:, i])
        self.data = data_array

    def deal_unsw_nb15(self, row):
        del row[-2]
        if row[3] == '-':
            row[3] = "else"
        return row

    def data_to_str(self, datas):
        new_data = []
        for data in datas:
            temp = ""
            for temp_data in data:
                temp = temp + str(temp_data)
                temp = temp + ","
            temp_list = list(temp)
            temp_list[-1] = "."
            temp = "".join(temp_list)
            temp = temp + "\n"
            new_data.append(temp)
        return new_data

    def deal_nsl_kdd(self, row):
        result = ""
        temp = row.split(",")
        # print(temp)
        del temp[-1]
        for temp_str in temp:
            result = result + temp_str
            result = result + ","
        temp_list = list(result)
        temp_list[-1] = "."
        result = "".join(temp_list)
        result = result + "\n"
        return result

    def del_headers_spqce(self, headers):
        result = []
        for header in headers:
            header_list = list(header)
            if header_list[0] == ' ':
                del header_list[0]
            result.append("".join(header_list))
        return result

    def get_CIC_IDS_2017(self):
        return 0

    def get_layer_attr(self, headers, name="UNSW-NB15"):
        result = []
        label_index = len(headers) - 1
        del_index = []
        if name == "UNSW-NB15":
            attrs = [
                ["srcip.*", "sport.*", "dstip.*", "dsport.*", "proto.*"],
                ["state.*", "dur.*", "sbytes.*", "dbytes.*", "sttl.*", "dttl.*", "sloss.*", "dloss.*", "service.*",
                 "sload.*", "dload.*", "spkts.*", "dpkts.*"],
                ["swin.*", "dwin.*", "stcpb.*", "dtcpb.*", "smeansz.*", "dmeansz.*", "trans_depth.*", "res_bdy_len.*"],
                ["sjit.*", "djit.*", "stime.*", "ltime.*", "sintpkt.*", "dintpkt.*", "tcprtt.*", "synack.*",
                 "ackdat.*"],
            ]
            label_index = 39
        elif name == "Tor":
            attrs = [
                ["Flow.*"],
                ["Fwd .*"],
                ["Bwd .*"],
                ["Active .*"],
                ["Idle .*"],
            ]
            del_index = [0, 2]
        elif name == "CIC-IDS-2017":
            attrs = [
                ["Flow .*"],
                ["Fwd .*"],
                ["Bwd .*"],
                ["Active .*"],
                ["Idle .*"],
            ]
        else:
            attrs = [
                ["Flow .*"],
                ["Fwd .*"],
                ["Bwd .*"],
                ["Active .*"],
                ["Idle .*"],
            ]
        for attr_list in attrs:
            layer_index = []
            for attr in attr_list:
                for i in range(len(headers)):
                    if re.match(attr, headers[i]) and (i not in del_index):
                        layer_index.append(i)
            result.append(layer_index)
        layer_index = []
        for i in range(len(headers)):
            if (not self.in_list(data=i, list=result)) and (not i == label_index) and (i not in del_index):
                layer_index.append(i)
        result.append(layer_index)
        return result, label_index

    def in_list(self, data, list):
        for temp in list:
            if data in temp:
                return True
        return False

    def normalization(self, temp_data):
        data = temp_data.tolist()
        data = self.str_to_double(data)
        min_data, max_data = min(data), max(data)
        if max_data == min_data:
            if max_data != 0:
                for i in range(len(data)):
                    data[i] = 1
        else:
            t = max_data - min_data
            for i in range(len(data)):
                data[i] = (data[i] - min_data) / t
        data = np.array(data)
        return data

    def str_to_double(self, data_list):
        result = []
        for data in data_list:
            result.append(float(data))
        return result

    def select_test_data(self, x, y, value=0, num=1000):
        res_x = []
        res_y = []
        count = 0
        for i in range(x.shape[0]):
            if y[i] == value or y[i] == str(value):
                res_x.append(x[i, :].tolist())
                res_y.append(y[i])
                count += 1
                if count == num:
                    break
        return np.array(res_x), np.array(res_y)
