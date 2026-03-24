from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

tf.random.set_seed(42)

data = load_breast_cancer()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
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

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

nn = Sequential([
    Dense(16, activation="relu", input_shape=(X_train_scaled.shape[1],)),
    Dense(1, activation="sigmoid")
])

nn.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

nn.fit(
    X_train_scaled,
    y_train,
    epochs=50,
    batch_size=16,
    verbose=0
)

dt_predictions = dt_constrained.predict(X_test)

nn_probabilities = nn.predict(X_test_scaled, verbose=0)
nn_predictions = (nn_probabilities > 0.5).astype(int).flatten()

dt_cm = confusion_matrix(y_test, dt_predictions)
nn_cm = confusion_matrix(y_test, nn_predictions)

print("Confusion Matrix for Constrained Decision Tree:")
print(dt_cm)

print("\nConfusion Matrix for Neural Network:")
print(nn_cm)

# COMMENTS:
# I would prefer the neural network for this task if the main goal is predictive
# performance, because it can capture more complex relationships in the data.
# One advantage of the constrained Decision Tree is that it is easy to interpret.
# One limitation of the constrained Decision Tree is that it may miss more complex
# patterns because of its restricted depth.
# One advantage of the Neural Network is that it can learn more complex patterns.
# One limitation of the Neural Network is that it is less interpretable and behaves
# more like a black box.