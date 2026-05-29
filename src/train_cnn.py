from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout,
    Input
)

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Dataset Paths

train_path = "dataset/train"
val_path = "dataset/val"

# Data Generators

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(
    rescale=1./255
)

# Train Generator

train_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# Validation Generator


val_generator = val_datagen.flow_from_directory(
    val_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

# CNN Model

model = Sequential([

    Input(shape=(224, 224, 3)),

    Conv2D(
        32,
        (3, 3),
        activation='relu'
    ),

    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(
        64,
        (3, 3),
        activation='relu'
    ),

    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(
        128,
        (3, 3),
        activation='relu'
    ),

    MaxPooling2D(pool_size=(2, 2)),

    Flatten(),

    Dense(64, activation='relu'),

    Dropout(0.5),

    Dense(4, activation='softmax')
])

# Compile Model

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train Model

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10
)

# Save Model

model.save("outputs/cnn_model.keras")

print("Model Saved Successfully")

# Accuracy Plot

plt.figure(figsize=(8, 5))

plt.plot(
    history.history['accuracy'],
    label='Training Accuracy'
)

plt.plot(
    history.history['val_accuracy'],
    label='Validation Accuracy'
)

plt.title("CNN Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.legend()

plt.savefig("outputs/cnn_accuracy.png")

plt.show()

# Loss Plot

plt.figure(figsize=(8, 5))

plt.plot(
    history.history['loss'],
    label='Training Loss'
)

plt.plot(
    history.history['val_loss'],
    label='Validation Loss'
)

plt.title("CNN Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.legend()

plt.savefig("outputs/cnn_loss.png")

plt.show()

# Predictions

val_generator.reset()

predictions = model.predict(val_generator)

predicted_classes = np.argmax(
    predictions,
    axis=1
)

true_classes = val_generator.classes

# Confusion Matrix

cm = confusion_matrix(
    true_classes,
    predicted_classes
)

plt.figure(figsize=(8, 6))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues'
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.savefig("outputs/confusion_matrix.png")

plt.show()

# Classification Report

class_labels = list(
    val_generator.class_indices.keys()
)

print("\nClassification Report:\n")

print(classification_report(
    true_classes,
    predicted_classes,
    target_names=class_labels
))