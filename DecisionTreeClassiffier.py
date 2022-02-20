from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np



class Node:

    def __init__(self, X, y, gini):
        self.X = X
        self.y = y
        self.gini = gini
        self.feature_index = 0
        self.threshold = 0
        self.left = None
        self.right = None


def make_terminal(group):
    class_vals = [row[-1] for row in group]
    return max(set(class_vals), key=class_vals.count)


class MyDecisionTreeClassifier:

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

    def split_data(self, X, y) -> tuple[int, int]:
        # test all the possible splits in O(N^2)
        # return index and threshold value
        best_idx, best_value, best_score, best_groups = \
            float('inf'), float('inf'), float('inf'), None
        # unite X and y
        X = [np.append(X[i], y[i]) for i in range(len(X))]
        classes = list(set(y))
        for idx in range(len(X[0]) - 1):
            for row in X:
                left, right = [], []
                for series in X:
                    if series[idx] < row[idx]:
                        left.append(series)
                    else:
                        right.append(series)
                gini = self.gini([left, right], classes)
                print(f'X{idx+1} < {round(row[idx], 3)} Gini={round(gini, 3)}')
                if gini < best_score:
                    best_idx, best_value, best_score, best_groups = idx, row[idx], gini, [left, right]
        return {'column': best_idx, 'value': best_value, 'groups': best_groups}

    def perform_split(self, node, depth):
        left, right = node['groups']
        del node['groups']
        if not left or not right:
            node['left'] = node['right'] = make_terminal(left + right)
            return  # stop building process
        if depth >= self.max_depth:
            node['left'] = make_terminal(left)
            node['right'] = make_terminal(right)
            return  # stop building process
        values_left = [row[:2] for row in left]
        classes_left = [row[-1] for row in left]
        node['left'] = self.split_data(values_left, classes_left)
        self.perform_split(node['left'], depth + 1)
        values_right = [row[:2] for row in right]
        classes_right = [row[-1] for row in right]
        node['right'] = self.split_data(values_right, classes_right)
        self.perform_split(node['right'], depth + 1)

    def build_tree(self, X, y, depth=0):
        # create a root node
        root = self.split_data(X, y)
        # recursively split until max depth is not exceed
        self.perform_split(root, depth)
        return root

    def fit(self, X, y):
        # basically wrapper for build tree

        pass

    def print_tree(self, node, depth=0):
        if isinstance(node, dict):
            # print(node)
            print(f"{depth * ' '}[X{node['column'] + 1} < {node['value']}]")
            self.print_tree(node['left'], depth + 1)
            self.print_tree(node['right'], depth + 1)
        else:
            print(f"{depth * ' '}[{node}]")

    def predict(self, X_test, X, y):
        # traverse the tree while there is left node
        # and return the predicted class for it,
        # note that X_test can be not only one example
        tree = self.build_tree(X, y, 1)
        result = []
        for row in X_test:
            result.append(self.perform_prediction(row, tree))
        return result

    def perform_prediction(self, row, node):
        if row[node['column']] < node['value']:
            if isinstance(node['left'], dict):
                return self.perform_prediction(row, node['left'])
            else:
                return node['left']
        else:
            if isinstance(node['right'], dict):
                return self.perform_prediction(row, node['right'])
            else:
                return node['right']


a = MyDecisionTreeClassifier(5)
X = [[2.771244718, 1.784783929],
     [1.728571309, 1.169761413],
     [3.678319846, 2.81281357],
     [3.961043357, 2.61995032],
     [2.999208922, 2.209014212],
     [7.497545867, 3.162953546],
     [9.00220326, 3.339047188],
     [7.444542326, 0.476683375],
     [10.12493903, 3.234550982],
     [6.642287351, 3.319983761]]
y = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]

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
iris = load_iris()
X = iris.data[:, :2]
y = iris.target
X, X_test, y, y_test = train_test_split(X, y, test_size=0.20)
print(X_test.shape)
tree = a.build_tree(X, y, 1)
a.print_tree(tree)
predictions = a.predict(X_test, X, y)
print(predictions)
print(sum(predictions == y_test) / len(y_test))
print(y_test)

