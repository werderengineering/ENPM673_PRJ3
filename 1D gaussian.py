import numpy as np

def initial_data():
    data_1 = np.random.normal(0, 2, (50, 1))
    data_2 = np.random.normal(3, 0.5, (50, 1))
    data_3 = np.random.normal(6, 3, (50, 1))
    # print("data1", data_1)
    # print("data2", data_2)
    # print("data3", data_3)
    data = np.concatenate((data_1, data_2, data_3), axis=0)
    return data

