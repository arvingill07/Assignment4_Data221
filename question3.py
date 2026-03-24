from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import numpy as np

data = load_breast_cancer()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

dt_constrained = DecisionTreeClassifier(
    criterion="entropy",
    max_depth=3,
    random_state=42
)
dt_constrained.fit(X_train, y_train)

train_accuracy = dt_constrained.score(X_train, y_train)
test_accuracy = dt_constrained.score(X_test, y_test)

print("Constrained Training Accuracy:", train_accuracy)
print("Constrained Test Accuracy:", test_accuracy)

importances = dt_constrained.feature_importances_
indices = np.argsort(importances)[-5:][::-1]

print("\nTop Five Most Important Features:")
for i in indices:
    print(data.feature_names[i], ":", importances[i])

# COMMENTS:
# Controlling model complexity helps reduce overfitting because the tree
# is prevented from becoming too deep and too specific to the training data.
# Feature importance improves interpretability because it shows which
# features have the greatest influence on the model's decisions.