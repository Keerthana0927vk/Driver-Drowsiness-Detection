import os
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Dataset Paths
train_path = "dataset/train"
val_path = "dataset/val"
test_path = "dataset/test"

# Data Augmentation + Rescaling
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.8,1.2]
)

# Only rescaling for validation & test
val_test_datagen = ImageDataGenerator(rescale=1./255)

# Train Generator
train_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=(224,224),
    batch_size=32,
    class_mode='categorical'
)

# Validation Generator
val_generator = val_test_datagen.flow_from_directory(
    val_path,
    target_size=(224,224),
    batch_size=32,
    class_mode='categorical'
)

# Test Generator
test_generator = val_test_datagen.flow_from_directory(
    test_path,
    target_size=(224,224),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

# Display class labels
print("\nClass Labels:")
print(train_generator.class_indices)

# Display Augmented Images
images, labels = next(train_generator)

plt.figure(figsize=(12,8))

for i in range(4):
    plt.subplot(2,2,i+1)
    plt.imshow(images[i])
    plt.title("Augmented Image")
    plt.axis("off")

plt.tight_layout()

# Save output image
plt.savefig("outputs/augmented_images.png")

plt.show()