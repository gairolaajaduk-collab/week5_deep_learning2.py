# ============================================================
# WEEK 5: DEEP LEARNING APPLICATION IN DATA SCIENCE
# Handwritten Digit Classification using Neural Network
# Dataset: MNIST
# Framework: TensorFlow / Keras
# ============================================================


# ------------------------------------------------------------
# 1. Import Libraries
# ------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf

from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Flatten,
    Dense,
    Dropout
)

from sklearn.metrics import (
    confusion_matrix,
    classification_report
)


# ------------------------------------------------------------
# 2. Display TensorFlow Version
# ------------------------------------------------------------

print("TensorFlow Version:")
print(tf.__version__)


# ------------------------------------------------------------
# 3. Load MNIST Dataset
# ------------------------------------------------------------

print("\nLoading MNIST dataset...")

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

print("Dataset loaded successfully.")

print("\nTraining images:", X_train.shape)
print("Training labels:", y_train.shape)

print("Testing images:", X_test.shape)
print("Testing labels:", y_test.shape)


# ------------------------------------------------------------
# 4. Display Sample Images
# ------------------------------------------------------------

plt.figure(figsize=(10, 5))

for i in range(10):

    plt.subplot(2, 5, i + 1)

    plt.imshow(X_train[i], cmap="gray")

    plt.title("Digit: " + str(y_train[i]))

    plt.axis("off")

plt.suptitle("Sample MNIST Images")

plt.tight_layout()

plt.show()


# ------------------------------------------------------------
# 5. Data Preprocessing
# ------------------------------------------------------------

# Pixel values are between 0 and 255.
# Convert them to values between 0 and 1.

X_train = X_train.astype("float32") / 255.0

X_test = X_test.astype("float32") / 255.0


print("\nData preprocessing completed.")

print("Minimum pixel value:", X_train.min())

print("Maximum pixel value:", X_train.max())


# ------------------------------------------------------------
# 6. Create Neural Network Model
# ------------------------------------------------------------

model = Sequential([

    # Convert 28 x 28 image into one-dimensional array
    Flatten(input_shape=(28, 28)),

    # First hidden layer
    Dense(128, activation="relu"),

    # Dropout helps reduce overfitting
    Dropout(0.2),

    # Second hidden layer
    Dense(64, activation="relu"),

    # Another dropout layer
    Dropout(0.2),

    # Output layer
    # 10 neurons because digits are 0 to 9
    Dense(10, activation="softmax")
])


# ------------------------------------------------------------
# 7. Display Model Architecture
# ------------------------------------------------------------

print("\nNeural Network Architecture:")

model.summary()


# ------------------------------------------------------------
# 8. Compile the Model
# ------------------------------------------------------------

model.compile(

    optimizer="adam",

    loss="sparse_categorical_crossentropy",

    metrics=["accuracy"]
)


print("\nModel compilation completed.")


# ------------------------------------------------------------
# 9. Train the Model
# ------------------------------------------------------------

print("\nStarting model training...")

history = model.fit(

    X_train,
    y_train,

    epochs=10,

    batch_size=32,

    validation_split=0.2,

    verbose=1
)


print("\nModel training completed.")


# ------------------------------------------------------------
# 10. Evaluate the Model
# ------------------------------------------------------------

print("\nEvaluating model on test data...")

test_loss, test_accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=0
)


print("\n========================================")
print("MODEL PERFORMANCE")
print("========================================")

print("Test Loss:", test_loss)

print("Test Accuracy:", test_accuracy)

print(
    "Test Accuracy Percentage:",
    round(test_accuracy * 100, 2),
    "%"
)


# ------------------------------------------------------------
# 11. Plot Training and Validation Accuracy
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.title("Training and Validation Accuracy")

plt.legend()

plt.show()


# ------------------------------------------------------------
# 12. Plot Training and Validation Loss
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.title("Training and Validation Loss")

plt.legend()

plt.show()


# ------------------------------------------------------------
# 13. Make Predictions
# ------------------------------------------------------------

print("\nMaking predictions...")

predictions = model.predict(X_test)

predicted_labels = np.argmax(
    predictions,
    axis=1
)


print("Predictions completed.")


# ------------------------------------------------------------
# 14. Classification Report
# ------------------------------------------------------------

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        predicted_labels
    )
)


# ------------------------------------------------------------
# 15. Confusion Matrix
# ------------------------------------------------------------

cm = confusion_matrix(
    y_test,
    predicted_labels
)


print("\nConfusion Matrix:")

print(cm)


plt.figure(figsize=(10, 8))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.xlabel("Predicted Digit")

plt.ylabel("Actual Digit")

plt.title("MNIST Confusion Matrix")

plt.show()


# ------------------------------------------------------------
# 16. Display Predictions
# ------------------------------------------------------------

plt.figure(figsize=(12, 6))

for i in range(10):

    plt.subplot(2, 5, i + 1)

    plt.imshow(
        X_test[i],
        cmap="gray"
    )

    plt.title(
        "Actual: "
        + str(y_test[i])
        + "\nPredicted: "
        + str(predicted_labels[i])
    )

    plt.axis("off")


plt.suptitle("Actual vs Predicted Digits")

plt.tight_layout()

plt.show()


# ------------------------------------------------------------
# 17. Predict a Single Image
# ------------------------------------------------------------

index = 0

single_image = X_test[index]

single_image_input = np.expand_dims(
    single_image,
    axis=0
)


prediction = model.predict(
    single_image_input
)


predicted_digit = np.argmax(
    prediction
)


print("\n========================================")
print("SINGLE IMAGE PREDICTION")
print("========================================")

print("Actual Digit:", y_test[index])

print("Predicted Digit:", predicted_digit)


# ------------------------------------------------------------
# 18. Prediction Probability
# ------------------------------------------------------------

print("\nPrediction Probabilities:")

for digit in range(10):

    probability = prediction[0][digit] * 100

    print(
        "Digit",
        digit,
        ":",
        round(probability, 2),
        "%"
    )


# ------------------------------------------------------------
# 19. Save the Trained Model
# ------------------------------------------------------------

model.save(
    "mnist_digit_classifier.keras"
)

print("\nTrained model saved as:")
print("mnist_digit_classifier.keras")


# ------------------------------------------------------------
# 20. Final Result
# ------------------------------------------------------------

print("\n========================================")
print("FINAL RESULT")
print("========================================")

print(
    "Test Accuracy:",
    round(test_accuracy * 100, 2),
    "%"
)

print(
    "Actual Digit:",
    y_test[index]
)

print(
    "Predicted Digit:",
    predicted_digit
)

print("========================================")
