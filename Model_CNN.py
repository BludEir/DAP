import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
import tensorflow as tf
from tensorflow.keras import layers, models, Input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import matplotlib.pyplot as plt


# Preprocess and normalize the image
def load_and_preprocess_image(image_path, target_size=(128, 128)):
    # Read the image in grayscale
    img = cv2.imread(f"{image_path}.png", cv2.IMREAD_GRAYSCALE)
    if img is None:  # Check if image is read correctly
        raise ValueError(f"Image not found: {image_path}")

    # Resize image to target size
    img = cv2.resize(img, target_size)

    # Normalize the image by standardizing (zero mean, unit variance)
    img = (img.astype(np.float32) - img.mean()) / img.std()

    # Apply Gaussian blur to reduce noise
    img = cv2.GaussianBlur(img, (5, 5), 0)

    # Canny edge detection to highlight boundaries
    median_intensity = np.median(img)
    lower_threshold = int(max(0, 0.7 * median_intensity * 255))
    upper_threshold = int(min(255, 1.3 * median_intensity * 255))

    img = cv2.Canny((img * 255).astype(np.uint8), lower_threshold, upper_threshold)

    # Normalize the edges to [0, 1] range
    img = img.astype(np.float32) / 255.0

    # Add the channel dimension (for grayscale, it will be 1 channel)
    img = np.expand_dims(img, axis=-1)

    return img


# Load the training dataset
def load_train():
    df = pd.read_csv(r"ADNI1_Complete_1Yr_1.5T.csv")
    images = []
    labels = []
    for _, row in df.iterrows():
        image_path = os.path.join("Middle Slice Train", str(row["Image Data ID"]))
        image = load_and_preprocess_image(image_path)
        images.append(image)

        # Assign labels based on the 'Group' column
        if row["Group"] == "AD":
            label = 0
        elif row["Group"] == "MCI":
            label = 1
        elif row["Group"] == "CN":
            label = 2
        labels.append(label)

    images = np.array(images)
    labels = np.array(labels)

    return images, labels


# Load the test dataset
def load_test(target_size=(128, 128)):
    df = pd.read_csv(r"ADNI1_Test.csv")
    images = []
    labels = []
    for _, row in df.iterrows():
        image_path = os.path.join("Middle Slice Test", str(row["Image Data ID"]))
        image = load_and_preprocess_image(image_path)
        images.append(image)

        # Assign labels based on the 'Group' column
        if row["Group"] == "AD":
            label = 0
        elif row["Group"] == "MCI":
            label = 1
        elif row["Group"] == "CN":
            label = 2
        labels.append(label)

    images = np.array(images)
    labels = np.array(labels)

    return images, labels


# Build the CNN model
def build_model(input_shape, num_classes):
    inputs = Input(shape=input_shape)

    # Convolutional layers
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # Flatten and fully connected layers
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)

    # Output layer
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs, outputs)

    return model


# Train the model
def train_model(X_train, X_test, Y_train, Y_test, input_shape):
    class_weights = class_weight.compute_class_weight(
        'balanced',
        classes=np.unique(np.argmax(Y_train, axis=1)),
        y=np.argmax(Y_train, axis=1)
    )
    class_weights = dict(enumerate(class_weights))

    model = build_model(input_shape, num_classes=3)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Data augmentation only on training data
    datagen = ImageDataGenerator(
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest'
    )
    datagen.fit(X_train)

    # Callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)
    checkpoint = ModelCheckpoint("Model/best_model.keras", monitor='val_accuracy', save_best_only=True, mode='max')

    # Training the model
    history = model.fit(datagen.flow(X_train, Y_train, batch_size=32),
                        epochs=50,
                        validation_data=(X_test, Y_test),
                        callbacks=[early_stopping, reduce_lr, checkpoint],
                        class_weight=class_weights
                        )

    test_loss, test_acc = model.evaluate(X_test, Y_test, verbose=2)
    print(f"\nTest accuracy: {test_acc:.4f}")

    return model, history


# Main function
def main():
    input_shape = (128, 128, 1)

    # Load data
    X_train, Y_train = load_train()
    X_test, Y_test = load_test()

    # One-hot encode the labels
    Y_train = to_categorical(Y_train, num_classes=3)
    Y_test = to_categorical(Y_test, num_classes=3)

    # Train the model
    model, history = train_model(X_train, X_test, Y_train, Y_test, input_shape)

    # Save the trained model
    model.save("Model/Model_CNN.keras")

    # Save model weights to H5
    model.save("Model/Model_CNN.h5")  # Save full model as .h5
    print("Model saved as full H5 file.")

    # Plot the training history
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
