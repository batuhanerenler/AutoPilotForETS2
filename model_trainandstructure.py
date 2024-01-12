import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, TimeDistributed, LSTM
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.optimizers import Adam

# Constants
DATA_DIR = 'ets2_data'  # Directory where your data is stored
IMG_WIDTH, IMG_HEIGHT = 224, 224  # Image dimensions
SEQUENCE_LENGTH = 5  # Number of frames in a sequence
NUM_CLASSES = 10  # Number of keyboard actions

# Load and preprocess data
def load_data(data_dir, img_width, img_height, sequence_length):
    sequences = []
    labels = []

    for seq_num in range(len(os.listdir(data_dir)) // sequence_length):
        sequence = []
        for frame_num in range(sequence_length):
            # Construct filename
            filename = f"image_{seq_num * sequence_length + frame_num}.png"
            filepath = os.path.join(data_dir, filename)

            # Load and preprocess image
            img = cv2.imread(filepath)
            img = cv2.resize(img, (img_width, img_height))
            sequence.append(img)

        # Load label (assuming labels are in 'key_logs.txt')
        with open(os.path.join(data_dir, "key_logs.txt"), "r") as file:
            lines = file.readlines()
            action = int(lines[seq_num * sequence_length].split(",")[1].strip())  # Example label extraction

        sequences.append(sequence)
        labels.append(action)

    return np.array(sequences), np.array(labels)

# Load the data
sequences, labels = load_data(DATA_DIR, IMG_WIDTH, IMG_HEIGHT, SEQUENCE_LENGTH)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(sequences, labels, test_size=0.2, random_state=42)

# Using EfficientNetB0 as a base model
base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(IMG_WIDTH, IMG_HEIGHT, 3))

# Freezing the base model layers
base_model.trainable = False

# Building the CNN-LSTM model
model = Sequential([
    TimeDistributed(base_model, input_shape=(SEQUENCE_LENGTH, IMG_WIDTH, IMG_HEIGHT, 3)),
    TimeDistributed(Flatten()),

    # LSTM layers
    LSTM(64, return_sequences=False),

    # Fully connected layers
    Dense(512, activation='relu'),
    Dropout(0.3),
    Dense(256, activation='relu'),
    Dropout(0.3),
    Dense(NUM_CLASSES, activation='softmax')
])

# Compile the model
optimizer = Adam(lr=0.0001)
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=4, validation_data=(X_test, y_test))

# Save the model
model.save('ets2_autonomous_driving_model.h5')

