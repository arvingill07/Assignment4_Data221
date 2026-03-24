from sklearn.datasets import load_breast_cancer
import numpy as np

data = load_breast_cancer()

X = data.data
y = data.target

print("Shape of X:", X.shape)
print("Shape of y:", y.shape)

class_counts = np.bincount(y)
print("Number of malignant samples:", class_counts[0])
print("Number of benign samples:", class_counts[1])

# COMMENTS:
# The dataset is slightly imbalanced because the number of benign samples
# is greater than the number of malignant samples.
# Class balance is important because if one class appears much more often,
# a classification model may become biased toward that majority class and
# perform worse on the minority class.