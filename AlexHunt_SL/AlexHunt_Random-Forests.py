import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools

# Load and prepare data
data = pd.read_csv('healthcare-dataset-stroke-data.csv')
data['bmi'] = data['bmi'].replace('N/A', np.nan).astype(float).fillna(data['bmi'].mean())
data = pd.get_dummies(data, columns=['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status'])
data.drop('id', axis=1, inplace=True)
X = data.drop('stroke', axis=1).values
y = data['stroke'].values

def calculate_entropy(y):
    _, class_indices = np.unique(y, return_inverse=True)
    probabilities = np.bincount(class_indices) / len(y)
    return -np.sum(probabilities * np.log2(probabilities + 1e-10))

def best_split(X, y):
    n_samples, n_features = X.shape
    best_feature, best_threshold, best_gain = None, None, -1
    parent_entropy = calculate_entropy(y)
    for feature_index in range(n_features):
        thresholds = np.unique(X[:, feature_index])
        for threshold in thresholds:
            left_indices = X[:, feature_index] <= threshold
            right_indices = X[:, feature_index] > threshold
            if np.sum(left_indices) == 0 or np.sum(right_indices) == 0:
                continue
            left_entropy = calculate_entropy(y[left_indices])
            right_entropy = calculate_entropy(y[right_indices])
            n_left, n_right = np.sum(left_indices), np.sum(right_indices)
            child_entropy = (n_left / n_samples) * left_entropy + (n_right / n_samples) * right_entropy
            information_gain = parent_entropy - child_entropy
            if information_gain > best_gain:
                best_gain = information_gain
                best_feature = feature_index
                best_threshold = threshold
    return best_feature, best_threshold

class DecisionNode:
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, *, value=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

class DecisionTree:
    def __init__(self, max_depth, n_features=None):
        self.max_depth = max_depth
        self.n_features = n_features
        self.root = None

    def build_tree(self, X, y):
        self.root = self.build_tree_helper(X, y, 0)

    def build_tree_helper(self, X, y, depth):
        if depth == self.max_depth or len(np.unique(y)) == 1:
            return DecisionNode(value=np.argmax(np.bincount(y)))

        if self.n_features is None:
            n_features = X.shape[1]
        else:
            n_features = self.n_features

        feature_indices = np.random.choice(X.shape[1], n_features, replace=False)
        best_feature, best_threshold = best_split(X[:, feature_indices], y)
        if best_feature is None:
            return DecisionNode(value=np.argmax(np.bincount(y)))

        left_indices = X[:, feature_indices[best_feature]] <= best_threshold
        right_indices = X[:, feature_indices[best_feature]] > best_threshold

        left_subtree = self.build_tree_helper(X[left_indices], y[left_indices], depth + 1)
        right_subtree = self.build_tree_helper(X[right_indices], y[right_indices], depth + 1)

        return DecisionNode(feature_index=feature_indices[best_feature], threshold=best_threshold, left=left_subtree, right=right_subtree)

    def predict(self, X):
        return np.array([self._predict_row(self.root, row) for row in X])

    def _predict_row(self, node, x):
        while node.value is None:
            if x[node.feature_index] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.value

class RandomForest:
    def __init__(self, n_trees=100, max_depth=10, n_features=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.n_features = n_features
        self.trees = []

    def fit(self, X, y):
        self.trees = []
        for _ in range(self.n_trees):
            tree = DecisionTree(max_depth=self.max_depth, n_features=self.n_features)
            X_sample, y_sample = self._bootstrap_sample(X, y)
            tree.build_tree(X_sample, y_sample)
            self.trees.append(tree)

    def predict(self, X):
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        return self._majority_vote(tree_preds)

    def _bootstrap_sample(self, X, y):
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, n_samples, replace=True)
        return X[indices], y[indices]

    def _majority_vote(self, predictions):
        majority_votes = np.apply_along_axis(lambda x: np.bincount(x, minlength=2).argmax(), axis=0, arr=predictions)
        return majority_votes

def calculate_accuracy(y_true, y_pred):
    correct_predictions = sum(1 for true, pred in zip(y_true, y_pred) if true == pred)
    accuracy = correct_predictions / len(y_true)
    return accuracy

def build_confusion_matrix(y_true, y_pred):
    tp = sum(1 for true, pred in zip(y_true, y_pred) if true == 1 and pred == 1)
    tn = sum(1 for true, pred in zip(y_true, y_pred) if true == 0 and pred == 0)
    fp = sum(1 for true, pred in zip(y_true, y_pred) if true == 0 and pred == 1)
    fn = sum(1 for true, pred in zip(y_true, y_pred) if true == 1 and pred == 0)
    return np.array([[tn, fp], [fn, tp]])

# Grid search for hyperparameters
tree_counts = [10]
max_depths = [5, 10, 15, 20, 25]
n_features_options = ['sqrt', 'log2']  # Options for n_features

results = []

for n_trees in tree_counts:
    for max_depth in max_depths:
        for n_features in n_features_options:
            if n_features == 'sqrt':
                n_features_value = int(np.sqrt(X.shape[1]))
            elif n_features == 'log2':
                n_features_value = int(np.log2(X.shape[1]))
            
            accuracies = []
            for _ in range(5):
                forest = RandomForest(n_trees=n_trees, max_depth=max_depth, n_features=n_features_value)
                forest.fit(X, y)
                predictions = forest.predict(X)
                accuracy = calculate_accuracy(y, predictions)
                accuracies.append(accuracy)

            confusion_mat = build_confusion_matrix(y, predictions)
            print("Confusion Matrix:")
            print(confusion_mat)
            
            avg_accuracy = np.mean(accuracies)
            results.append((n_features, max_depth, avg_accuracy))
            print(f"Feature type: {n_features}, Depth: {max_depth}, Average Accuracy: {avg_accuracy}")

# Plotting the results
plt.figure(figsize=(12, 8))
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']  # Add more colors if you have more combinations
marker = itertools.cycle(('+', 'o', '*', 'x', 's'))  # Different markers for visibility

for n_features in n_features_options:
    feature_depths = [depth for feature, depth, acc in results if feature == n_features]
    feature_accuracies = [acc for feature, depth, acc in results if feature == n_features]
    plt.plot(feature_depths, feature_accuracies, marker=next(marker), color=colors.pop(0), label=f'Feature: {n_features}')

plt.title('Accuracy vs. Feature Type and Tree Depth in Random Forest', fontsize=18)
plt.xlabel('Tree Depth', fontsize=14)
plt.ylabel('Average Accuracy', fontsize=14)
plt.xticks(max_depths, fontsize=12)
plt.yticks(fontsize=12)
plt.legend(loc='lower right', fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.show()