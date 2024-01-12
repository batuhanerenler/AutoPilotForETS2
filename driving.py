import cv2
import numpy as np
import pygetwindow as gw
from PIL import ImageGrab
import tensorflow as tf
import time
import pyautogui

# Constants
MODEL_PATH = 'ets2_autonomous_driving_model.h5'
IMG_WIDTH, IMG_HEIGHT = 224, 224
SEQUENCE_LENGTH = 5
WINDOW_TITLE = "Euro Truck Simulator 2"
SLEEP_TIME = 0.1

# Load the trained model
model = tf.keras.models.load_model(MODEL_PATH)

# Define action mapping based on your model's training
ACTION_MAPPING = {
    0: 'w',  # Forward
    1: 'a',  # Left
    2: 'd',  # Right
    # Add more mappings as per your model's output
}

# Function to preprocess the captured frame
def preprocess_frame(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (IMG_WIDTH, IMG_HEIGHT))
    frame = frame / 255.0
    return frame

# Function to capture the screen
def capture_screen(window_title):
    game_window = gw.getWindowsWithTitle(window_title)[0]
    game_window.activate()
    x, y, width, height = game_window.left, game_window.top, game_window.width, game_window.height
    screen = np.array(ImageGrab.grab(bbox=(x, y, x + width, y + height)))
    return screen

# Function to make a prediction and return the corresponding action
def make_prediction(sequence):
    sequence = np.array([sequence])
    prediction = model.predict(sequence)
    action_index = np.argmax(prediction, axis=1)[0]
    return ACTION_MAPPING.get(action_index, None)  # Returns None if index not in mapping

# Main loop to capture screen and make predictions
current_sequence = []
while True:
    screen = capture_screen(WINDOW_TITLE)
    preprocessed_frame = preprocess_frame(screen)
    current_sequence.append(preprocessed_frame)

    if len(current_sequence) == SEQUENCE_LENGTH:
        action = make_prediction(current_sequence)
        if action:
            print(f"Performing action: {action}")
            pyautogui.press(action)  # Send the key press to the game

        current_sequence.pop(0)

    time.sleep(SLEEP_TIME)
