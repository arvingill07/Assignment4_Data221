from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

data = load_breast_cancer()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

dt = DecisionTreeClassifier(criterion="entropy", random_state=42)
dt.fit(X_train, y_train)

train_accuracy = dt.score(X_train, y_train)
test_accuracy = dt.score(X_test, y_test)

print("Training Accuracy:", train_accuracy)
print("Test Accuracy:", test_accuracy)

# COMMENTS:
# Entropy measures the uncertainty or impurity in a node.
# In a decision tree, the model chooses splits that reduce entropy as much as possible.
# If the training accuracy is much higher than the test accuracy,
# that suggests overfitting.
# If the two accuracies are similar, that suggests better generalization.
# In this model results suggest slight overfitting.