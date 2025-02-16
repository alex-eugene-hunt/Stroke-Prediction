# Alex Hunt (113536050)
# (CS-5033-001) Machine Learning Fundamentals (Spring 2024) with Professor Diochnos
# SL Project - Decision Trees - Testing the effictiveness of different tree depths

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load and prepare data
data = pd.read_csv('healthcare-dataset-stroke-data.csv')
data['bmi'] = data['bmi'].replace('N/A', np.nan).astype(float).fillna(data['bmi'].mean())
data = pd.get_dummies(data, columns=['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status'])
data.drop('id', axis=1, inplace=True)
X = data.drop('stroke', axis=1).values
y = data['stroke'].values

class DecisionNode:
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, *, value=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

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

# Modify build_tree function to accept a list of depths to test
def build_tree(X, y, depths=[10]):
    accuracies = []
    sensitivities = []
    specificities = []
    for max_depth in depths:
        print("Tree Depth:", max_depth)
        root = build_tree_helper(X, y, max_depth=max_depth)
        predictions = [predict(root, x) for x in X]
        accuracy = calculate_accuracy(y, predictions)
        conf_matrix = build_confusion_matrix(y, predictions)
        accuracies.append(accuracy)
        sensitivity = conf_matrix[1, 1] / (conf_matrix[1, 1] + conf_matrix[1, 0])
        sensitivities.append(sensitivity)
        specificity = conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[0, 1])
        specificities.append(specificity)
        print("Accuracy:", accuracy)
        print("Confusion Matrix:")
        print(conf_matrix)
        print()

    # Plot accuracies, sensitivities, and specificities against depths
    plt.figure(figsize=(10, 6))
    plt.bar(depths, accuracies, color='blue', width=0.5, label='Accuracy')
    plt.bar(np.array(depths) + 0.3, sensitivities, color='red', width=0.5, label='Sensitivity (True Positive Rate)')
    #plt.bar(np.array(depths) + 0.6, specificities, color='orange', width=0.333, label='Specificity (True Negative Rate)')
    plt.title('Performance Metrics vs. Tree Depth', fontsize=18)
    plt.xlabel('Tree Depth', fontsize=14)
    plt.ylabel('Score', fontsize=14)
    plt.xticks(depths, fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(loc='lower right', fontsize=12)  # Position legend in the bottom right
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Helper function to build a tree with a specific depth
def build_tree_helper(X, y, depth=0, max_depth=10):
    if depth == max_depth or len(np.unique(y)) == 1:
        return DecisionNode(value=np.argmax(np.bincount(y)))

    feature_index, threshold = best_split(X, y)
    if feature_index is None:
        return DecisionNode(value=np.argmax(np.bincount(y)))

    left_indices = X[:, feature_index] <= threshold
    right_indices = X[:, feature_index] > threshold

    left_subtree = build_tree_helper(X[left_indices], y[left_indices], depth + 1, max_depth)
    right_subtree = build_tree_helper(X[right_indices], y[right_indices], depth + 1, max_depth)

    return DecisionNode(feature_index=feature_index, threshold=threshold, left=left_subtree, right=right_subtree)

def predict(node, x):
    while node.value is None:
        if x[node.feature_index] <= node.threshold:
            node = node.left
        else:
            node = node.right
    return node.value

# Evaluate the model
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

# Define depths to test
depths_to_test = [5, 10, 15, 20, 25]

# Build trees for different depths and print results
build_tree(X, y, depths_to_test)