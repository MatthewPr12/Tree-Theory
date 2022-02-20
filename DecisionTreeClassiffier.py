# pylint: disable=invalid-name
"""
Decision Tree Classifier module
"""
from sklearn.datasets import load_iris  # pylint: disable=import-error
from sklearn.model_selection import train_test_split  # pylint: disable=import-error
import numpy as np


class Node:
    """
    Node of the tree
    """
    def __init__(self, train_ds, train_classes, gini, feature_index, threshold, left, right):
        self.X = train_ds
        self.y = train_classes
        self.gini = gini
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right


def make_terminal(group):
    """
    make the node leaf
    :param group:
    :return:
    """
    class_vals = [row[-1] for row in group]
    return max(set(class_vals), key=class_vals.count)


class MyDecisionTreeClassifier:
    """
    tree-class
    """
    def __init__(self, max_depth):
        self.max_depth = max_depth

    def gini(self, groups, classes):
        """
        A Gini score gives an idea of how good a split is by how mixed the
        classes are in the two groups created by the split.

        A perfect separation results in a Gini score of 0,
        whereas the worst case split that results in 50/50
        classes in each group result in a Gini score of 0.5
        (for a 2 class problem).
        """
        num_of_rows = sum([len(group) for group in groups])
        gini_index = 0
        for group in groups:
            size = len(group)
            if size == 0:
                continue
            score = 0
            for class_val in classes:
                p = [row[-1] for row in group].count(class_val) / size
                score += p ** 2
            gini_index += (1.0 - score) * (size / num_of_rows)
        return gini_index

    # def eval_split(self, idx, value, X, y):
    #     left, right = [], []
    #     for row in X:

    def split_data(self, all_vals, classes_vals) -> tuple[int, int]:
        """
        split dataset by all columns and values in those columns
        :param all_vals:
        :param classes_vals:
        :return:
        """
        # test all the possible splits in O(N^2)
        # return index and threshold value
        best_idx, best_value, best_score, best_groups = \
            float('inf'), float('inf'), float('inf'), None
        # unite X and y
        all_vals = [np.append(all_vals[i], classes_vals[i]) for i in range(len(all_vals))]
        classes = list(set(classes_vals))
        for idx in range(len(all_vals[0]) - 1):
            for row in all_vals:
                left, right = [], []
                for series in all_vals:
                    if series[idx] < row[idx]:
                        left.append(series)
                    else:
                        right.append(series)
                gini = self.gini([left, right], classes)
                # print(f'X{idx + 1} < {round(row[idx], 3)} Gini={round(gini, 3)}')  # to see the ginis
                if gini < best_score:
                    best_idx, best_value, best_score, best_group_l,\
                    best_group_r = idx, row[idx], gini, left, right
        return Node(all_vals, classes_vals, best_score, best_idx,
                    best_value, best_group_l, best_group_r)
        # {'column': best_idx, 'value': best_value, 'groups': best_groups},

    def perform_split(self, node, depth):
        """
        do the split
        :param node:
        :param depth:
        :return:
        """
        left, right = node.left, node.right
        # del node['groups']
        if not left or not right:
            node.left = node.right = make_terminal(left + right)
            return  # stop building process
        if depth >= self.max_depth:
            node.left = make_terminal(left)
            node.right = make_terminal(right)
            return  # stop building process
        values_left = [row[:2] for row in left]
        classes_left = [row[-1] for row in left]
        node.left = self.split_data(values_left, classes_left)
        self.perform_split(node.left, depth + 1)
        values_right = [row[:2] for row in right]
        classes_right = [row[-1] for row in right]
        node.right = self.split_data(values_right, classes_right)
        self.perform_split(node.right, depth + 1)

    def build_tree(self, all_vals, classes_vals, depth=0):
        """
        build tree based on splits
        :param all_vals:
        :param classes_vals:
        :param depth:
        :return:
        """
        # create a root node
        root = self.split_data(all_vals, classes_vals)
        # recursively split until max depth is not exceed
        self.perform_split(root, depth)
        return root

    def print_tree(self, node, depth=0):
        """
        show the tree
        :param node:
        :param depth:
        :return:
        """
        if isinstance(node, Node):
            # print(node)
            print(f"{depth * ' '}[X{node.feature_index + 1} < {node.threshold}]")
            self.print_tree(node.left, depth + 1)
            self.print_tree(node.right, depth + 1)
        else:
            print(f"{depth * ' '}[{node}]")

    def predict(self, test_ds, train_ds, train_classes):
        """
        traverse the tree while there is left node
        and return the predicted class for it,
        note that test_ds can be not only one example
        :param test_ds:
        :param train_ds:
        :param train_classes:
        :return:
        """
        my_tree = self.build_tree(train_ds, train_classes, 1)
        result = []
        for row in test_ds:
            result.append(self.perform_prediction(row, my_tree))
        return result

    def perform_prediction(self, row, node):
        """
        predict the class
        :param row:
        :param node:
        :return:
        """
        if row[node.feature_index] < node.threshold:
            if isinstance(node.left, Node):
                return self.perform_prediction(row, node.left)
            return node.left
        if isinstance(node.right, Node):
            return self.perform_prediction(row, node.right)
        return node.right


if __name__ == '__main__':
    my_classifier = MyDecisionTreeClassifier(5)
    iris = load_iris()
    X = iris.data[:, :2]
    y = iris.target
    X, X_test, y, y_test = train_test_split(X, y, test_size=0.20)
    tree = my_classifier.build_tree(X, y, 1)
    my_classifier.print_tree(tree)
    predictions = my_classifier.predict(X_test, X, y)
    print(sum(predictions == y_test) / len(y_test))
    # X = [[2.771244718, 1.784783929],
    #      [1.728571309, 1.169761413],
    #      [3.678319846, 2.81281357],
    #      [3.961043357, 2.61995032],
    #      [2.999208922, 2.209014212],
    #      [7.497545867, 3.162953546],
    #      [9.00220326, 3.339047188],
    #      [7.444542326, 0.476683375],
    #      [10.12493903, 3.234550982],
    #      [6.642287351, 3.319983761]]
    # y = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]

    # dataset = [[2.771244718, 1.784783929, 0],
    #            [1.728571309, 1.169761413, 0],
    #            [3.678319846, 2.81281357, 0],
    #            [3.961043357, 2.61995032, 0],
    #            [2.999208922, 2.209014212, 0],
    #            [7.497545867, 3.162953546, 1],
    #            [9.00220326, 3.339047188, 1],
    #            [7.444542326, 0.476683375, 1],
    #            [10.12493903, 3.234550982, 1],
    #            [6.642287351, 3.319983761, 1]]
    #
    #  predict with a stump
    # stump = {'column': 0, 'right': 1, 'value': 6.642287351, 'left': 0}
    # for row in dataset:
    #     prediction = a.perform_prediction(row, stump)
    #     print('Expected=%d, Got=%d' % (row[-1], prediction))
    # print(a.gini([[[1, 1], [1, 0]], [[1, 1], [1, 0]]], [0, 1]))
    # print(a.gini([[[1, 0], [1, 0]], [[1, 1], [1, 1]]], [0, 1]))
    # print(a.split_data(X, y))
    # X_test = [[2.171244718, 1.084783929],
    #      [2.728571309, 1.069761413],
    #      [2.678319846, 4.81281357],
    #      [3.061043357, 1.61995032],
    #      [2.099208922, 1.209014212],
    #      [4.497545867, 3.962953546],
    #      [4.00220326, 2.339047188],
    #      [7.044542326, 2.476683375],
    #      [10.12493903, 3.234550982],
    #      [3.642287351, 1.319983761]]
