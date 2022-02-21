import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def read_data():
    file = pd.read_csv('dataset.csv')
    result_kruskal = {}
    result_prim = {}
    for index, row in file.iterrows():
        row = [float(x) for x in list(row)]
        if (row[3], row[4]) not in list(result_prim.keys()):
            result_prim[(row[3], row[4])] = row[2:]
            result_kruskal[(row[3], row[4])] = [row[1]] + row[3:]
    return list(result_kruskal.values()), list(result_prim.values())


def visual(data1, data2):
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
    ax.view_init(15, 200)
    plt.show()


visual(*read_data())
