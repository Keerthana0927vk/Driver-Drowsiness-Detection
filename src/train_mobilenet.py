from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    GlobalAveragePooling2D
)

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# =========================
# Dataset Paths
# =========================

train_path = "dataset/train"
val_path = "dataset/val"

# =========================
# Data Generators
# =========================

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(
    rescale=1./255
)

# =========================
# Train Generator
# =========================

train_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# =========================
# Validation Generator
# =========================

val_generator = val_datagen.flow_from_directory(
    val_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

# =========================
# Load MobileNetV2
# =========================

base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)

# Freeze Base Model
base_model.trainable = False

# =========================
# Build Model
# =========================

model = Sequential([

    base_model,

    GlobalAveragePooling2D(),

    Dense(128, activation='relu'),

    Dropout(0.5),

    Dense(4, activation='softmax')
])

# =========================
# Compile Model
# =========================

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# =========================
# Train Model
# =========================

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=5
)

# =========================
# Save Model
# =========================

model.save("models/mobilenet_model.keras")

print("MobileNetV2 Model Saved Successfully")

# =========================
# Accuracy Plot
# =========================

plt.figure(figsize=(8,5))

plt.plot(
    history.history['accuracy'],
    label='Training Accuracy'
)

plt.plot(
    history.history['val_accuracy'],
    label='Validation Accuracy'
)

plt.title("MobileNetV2 Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.legend()

plt.savefig("outputs/mobilenet_accuracy.png")

plt.show()

# =========================
# Loss Plot
# =========================

plt.figure(figsize=(8,5))

plt.plot(
    history.history['loss'],
    label='Training Loss'
)

plt.plot(
    history.history['val_loss'],
    label='Validation Loss'
)

plt.title("MobileNetV2 Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.legend()

plt.savefig("outputs/mobilenet_loss.png")

plt.show()

# =========================
# Predictions
# =========================

val_generator.reset()

predictions = model.predict(val_generator)

predicted_classes = np.argmax(
    predictions,
    axis=1
)

true_classes = val_generator.classes

# =========================
# Confusion Matrix
# =========================

cm = confusion_matrix(
    true_classes,
    predicted_classes
)

plt.figure(figsize=(8,6))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Greens'
)

plt.title("MobileNetV2 Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.savefig("outputs/mobilenet_confusion_matrix.png")

plt.show()

# =========================
# Classification Report
# =========================

class_labels = list(
    val_generator.class_indices.keys()
)

print("\nClassification Report:\n")

print(classification_report(
    true_classes,
    predicted_classes,
    target_names=class_labels
))