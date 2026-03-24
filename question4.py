from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
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

history = nn.fit(
    X_train_scaled,
    y_train,
    epochs=50,
    batch_size=16,
    verbose=0
)

train_loss, train_accuracy = nn.evaluate(X_train_scaled, y_train, verbose=0)
test_loss, test_accuracy = nn.evaluate(X_test_scaled, y_test, verbose=0)

print("Neural Network Training Accuracy:", train_accuracy)
print("Neural Network Test Accuracy:", test_accuracy)

# COMMENTS:
# Feature scaling is necessary for neural networks because the optimization
# process works better when all input features are on similar numeric scales.
# Without scaling, features with larger values can dominate the learning process
# and make training slower or less stable.
# An epoch represents one complete pass through the entire training dataset
# during neural network training.