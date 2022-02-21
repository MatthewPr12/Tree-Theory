import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def read_data(max_vertex):
    """

    :param max_vertex:
    :return:
    """
    file = pd.read_csv('dataset.csv')
    result_kruskal = []
    result_prim = []
    possib_repeat = int(100/int(max_vertex//5))
    for index, row in file.iterrows():
        row = [float(x) for x in list(row)]
        if row[3] > max_vertex or row[4] * 100 % possib_repeat != 0:
            continue
        result_prim.append(row[2:])
        result_kruskal.append([row[1]] + row[3:])
    return result_kruskal, result_prim


def visual(data1, data2):
    """

    :param data1:
    :param data2:
    :return:
    """
    data1 = np.array(data1)
    data2 = np.array(data2)
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(data1[:, 2], data1[:, 1], data1[:, 0], label='kruskal')
    ax.scatter(data2[:, 2], data2[:, 1], data2[:, 0], label='prim')
    ax.set_xlabel("Completeness")
    ax.set_ylabel("Num of nodes (N)")
    ax.set_zlabel("Time (s)")
    ax.legend(loc="best")
    ax.view_init(15, 225)
    plt.show()


if __name__ == "__main__":
    while True:
        print('Enter number of nodes (5<=nodes<=500)')
        num = input('>>> ')
        if num.isdigit() and 5 <= int(num) <= 500:
            num = int(num)
            break
        else:
            print('number (type: int)')
    visual(*read_data(num))
