# Alex Hunt (113536050)
# (CS-5033-001) Machine Learning Fundamentals (Spring 2024) with Professor Diochnos
# SL Project - Novelty - Creating a UI to see if a user is likely of a stroke

import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import messagebox, ttk

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

def submit_form():
    # Define a template for input data based on how the model was trained
    # Here, you must manually define the length and positions for one-hot encoded features
    # For example, let's assume the full feature set size is 20 for demonstration
    input_template = np.zeros(22)
    input_positions = {
        'Age': 0,
        'Gender':(1, 3),
        'Hypertension': (3, 5),  # Example range for binary categories (No, Yes)
        'Heart Disease': (5, 7),  # Continuing example range
        'Ever Married': (7, 9),
        'Work Type': (9, 14),  # Assuming five categories
        'Residence Type': (14, 16),
        'Avg Glucose Level': 16,
        'BMI': 17,
        'Smoking Status': (18, 22)  # Assuming four categories
    }

    # Gather input data from GUI and encode properly
    for entry, values in zip(entries, feature_info):
        if isinstance(values, list):  # Categorical features
            # Find the range in the template
            pos_range = input_positions[feature_names[entries.index(entry)]]
            # Set the appropriate index within the range to 1
            input_template[pos_range[0] + values.index(entry.get())] = 1
        else:
            # Directly place numerical values
            pos = input_positions[feature_names[entries.index(entry)]]
            input_template[pos] = float(entry.get())

    # Prediction
    try:
        prediction = predict(tree_root, input_template)
        message = f"You are {'likely' if prediction == 1 else 'unlikely'} to have a stroke."
        messagebox.showinfo("Prediction", message)
    except Exception as e:
        messagebox.showerror("Prediction Error", str(e))

# Feature names in order they appear in GUI
feature_names = ['Age', 'Gender', 'Hypertension', 'Heart Disease', 'Ever Married', 'Work Type', 'Residence Type', 'Avg Glucose Level', 'BMI', 'Smoking Status']

# Build decision tree model
tree_root = build_tree_helper(X, y, max_depth=10)

# GUI setup
root = tk.Tk()
root.title("Stroke Prediction")

# Define features and their options if they are categorical
features = {
    'Age': None,
    'Gender': ['Male', 'Female'],
    'Hypertension': ['No', 'Yes'],
    'Heart Disease': ['No', 'Yes'],
    'Ever Married': ['No', 'Yes'],
    'Work Type': ['Govt_job', 'Never_worked', 'Private', 'Self-employed', 'children'],
    'Residence Type': ['Rural', 'Urban'],
    'Avg Glucose Level': None,
    'BMI': None,
    'Smoking Status': ['Unknown', 'Never smoked', 'Formerly smoked', 'Smokes']
}
entries = []
feature_info = []

for feature, options in features.items():
    row = tk.Frame(root)
    row.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
    label = tk.Label(row, width=22, text=feature + ":", anchor='w')
    label.pack(side=tk.LEFT)
    if options:
        entry = ttk.Combobox(row, values=options)
        entry.current(0)
    else:
        entry = tk.Entry(row)
    entry.pack(side=tk.RIGHT, expand=tk.YES, fill=tk.X)
    entries.append(entry)
    feature_info.append(options)

button = tk.Button(root, text='Submit', command=submit_form)
button.pack(side=tk.RIGHT, padx=5, pady=5)

root.mainloop()