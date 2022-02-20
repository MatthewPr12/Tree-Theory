from sklearn.datasets import load_iris

iris = load_iris()
X = iris.data[:, :2]
y = iris.target


class Node:

    def __init__(self, X, y, gini):
        self.X = X
        self.y = y
        self.gini = gini
        self.feature_index = 0
        self.threshold = 0
        self.left = None
        self.right = None


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
        X = [X[i]+[y[i]] for i in range(len(X))]
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
                print('X%d < %.3f Gini=%.3f' % ((idx + 1), row[idx], gini))
                if gini < best_score:
                    best_idx, best_value, best_score, best_groups = idx, row[idx], gini, [left, right]
        return best_idx, best_value

    def build_tree(self, X, y, depth=0):
        # create a root node

        # recursively split until max depth is not exeeced

        pass

    def fit(self, X, y):
        # basically wrapper for build tree

        pass

    def predict(self, X_test):
        # traverse the tree while there is left node
        # and return the predicted class for it,
        # note that X_test can be not only one example

        pass


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
print(a.gini([[[1, 1], [1, 0]], [[1, 1], [1, 0]]], [0, 1]))
print(a.gini([[[1, 0], [1, 0]], [[1, 1], [1, 1]]], [0, 1]))
print(a.split_data(X, y))
