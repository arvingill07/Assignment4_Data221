from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras import models, layers

(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

X_train = X_train / 255.0
X_test = X_test / 255.0

X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

cnn = models.Sequential([
    layers.Input(shape=(28, 28, 1)),
    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(10, activation="softmax")
])

cnn.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

cnn.fit(X_train, y_train, epochs=15)

test_loss, test_accuracy = cnn.evaluate(X_test, y_test)
print("CNN Test Accuracy:", test_accuracy)

# COMMENTS:
# CNNs are generally preferred over fully connected networks for image data
# because they preserve spatial structure and learn local visual patterns.
# In this task, the convolution layer is learning useful image features
# such as edges, textures, and simple shapes.