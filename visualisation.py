import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import re


def read_data(max_vertex, plot=False):
    """
    read information from csv file
    :param plot: bool
    :param max_vertex: int
    :return: tuple
    """
    file = pd.read_csv('dataset.csv')
    result_kruskal = []
    result_prim = []
    if plot:
        possib_repeat = int(100 / int(max_vertex // 5))
    else:
        possib_repeat = 1
    for index, row in file.iterrows():
        row = [float(x) for x in list(row)]
        if row[3] > max_vertex or row[4] * 100 % possib_repeat != 0:
            continue
        result_prim.append(row[2:])
        result_kruskal.append([row[1]] + row[3:])
    return result_kruskal, result_prim


def visual(data1, data2):
    """
    create a 3d plot by data1 and data2
    :param data1: list
    :param data2: list
    :return: None
    """
    data1 = np.array(data1)
    data2 = np.array(data2)
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(data1[:, 2], data1[:, 1], data1[:, 0], label='kruskal', alpha=.2)
    ax.scatter(data2[:, 2], data2[:, 1], data2[:, 0], label='prim', alpha=.2)
    ax.set_xlabel("Completeness")
    ax.set_ylabel("Num of nodes (N)")
    ax.set_zlabel("Time (s)")
    ax.legend(loc="best")
    ax.view_init(15, 225)
    plt.show()


def d2_visual(data1, data2, possibility):
    """
    create a 2d plot by data1 and data2
    :param data1: list
    :param data2: list
    :param possibility: float
    :return: None
    """
    new_data1, new_data2 = [], []
    for item in data1:
        if item[2] == possibility:
            new_data1.append(item)
    for item in data2:
        if item[2] == possibility:
            new_data2.append(item)

    data1 = np.array(new_data1)
    data2 = np.array(new_data2)
    # data1_mid_point = data1[len(data1)//2]
    # data2_mid_point = data2[len(data2)//2]
    # data1_end_point = data1[len(data1) - 1]
    # data2_end_point = data2[len(data2) - 1]
    # plt.annotate('bla', data2_mid_point, data1_mid_point, data2_end_point, data1_end_point)
    # plt.plot(data1_mid_point, data2_mid_point, data1_end_point, data2_end_point, 'ro')
    plt.plot(data1[:, 1], data1[:, 0], label='kruskal')
    plt.plot(data2[:, 1], data2[:, 0], label='prim')
    plt.xlabel("Num of nodes (N)")
    plt.ylabel("Time (s)")
    plt.legend(loc="best")
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

    done = False
    while True:
        if done:
            break
        print('3d or 2d?')
        plot = input('>>> ')
        if re.match(r"^[3зЗ][дДDd]?$", plot):
            visual(*read_data(num))
            break
        elif re.match(r"^2[дДDd]?$", plot):
            while True:
                print('Possibility є [0.01, 1.0], enter one number')
                try:
                    possibility = float(input('>>> '))
                    if 0.01 <= possibility <= 1.0:
                        d2_visual(*read_data(num), possibility)
                        done = True
                        break
                except ValueError:
                    print('Possibility є [0.01, 1.0]')
