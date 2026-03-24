from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras import models, layers
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

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

y_pred = cnn.predict(X_test)
y_pred_labels = np.argmax(y_pred, axis=1)

cm = confusion_matrix(y_test, y_pred_labels)
print("Confusion Matrix:")
print(cm)

class_names = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

misclassified = np.where(y_pred_labels != y_test)[0]

for i in range(3):
    idx = misclassified[i]
    plt.imshow(X_test[idx].reshape(28, 28), cmap="gray")
    plt.title(
        "True: " + class_names[y_test[idx]] +
        ", Predicted: " + class_names[y_pred_labels[idx]]
    )
    plt.axis("off")
    plt.show()

# COMMENTS:
# One pattern in the misclassifications is that visually similar clothing items
# are sometimes confused, such as shirts, coats, and pullovers.
# One realistic way to improve CNN performance is to use a deeper network
# or apply more tuning, such as adjusting epochs or adding more layers.