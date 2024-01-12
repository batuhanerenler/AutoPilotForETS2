import cv2
import numpy as np
import time
from PIL import ImageGrab
from pynput import keyboard
import pygetwindow as gw
import os
import threading

# Constants
data_folder = "ets2_data"
sequence_length = 5  # Number of frames in a sequence
capture_interval = 0.1  # Time interval between captures in seconds
window_title = "Euro Truck Simulator 2"  # Adjust if necessary

# Ensure data folder exists
if not os.path.exists(data_folder):
    os.makedirs(data_folder)

# Global variables
current_sequence = []
key_presses = []
capturing = False
sequence_counter = 0

# Function to capture the screen and apply pre-processing
def capture_screen(window_title):
    game_window = gw.getWindowsWithTitle(window_title)[0]
    game_window.activate()
    x, y, width, height = game_window.left, game_window.top, game_window.width, game_window.height
    screen = np.array(ImageGrab.grab(bbox=(x, y, x + width, y + height)))
    screen = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)
    screen = cv2.resize(screen, (224, 224))  # Resize for the model
    screen = screen / 255.0  # Normalize
    return screen

# Function to log keyboard inputs
def on_press(key):
    global current_sequence, key_presses, capturing, sequence_counter
    if capturing:
        key_char = key.char if hasattr(key, 'char') else str(key)
        key_presses.append((time.time(), key_char))

# Function to handle start/stop capture
def on_release(key):
    global capturing
    if key == keyboard.Key.page_up:
        capturing = True
        print("Data collection started.")
    elif key == keyboard.Key.page_down:
        capturing = False
        print("Data collection stopped.")
        return False  # Stop the listener

# Function to save a sequence
def save_sequence(sequence, seq_counter):
    sequence_folder = os.path.join(data_folder, f"sequence_{seq_counter}")
    os.makedirs(sequence_folder, exist_ok=True)

    label_file_path = os.path.join(sequence_folder, "labels.txt")
    with open(label_file_path, "w") as label_file:
        for i, frame in enumerate(sequence):
            screenshot_path = os.path.join(sequence_folder, f"frame_{i}.png")
            cv2.imwrite(screenshot_path, frame * 255)  # Convert back to original scale
            label_file.write(f"{i}, {key_presses[i][1]}\n")

# Modified data collection loop
def start_data_collection():
    global current_sequence, key_presses, sequence_counter
    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        while listener.running:
            if capturing and len(current_sequence) < sequence_length:
                screen = capture_screen(window_title)
                current_sequence.append(screen)
                time.sleep(capture_interval)
            elif capturing and len(current_sequence) == sequence_length:
                save_sequence(current_sequence, sequence_counter)
                current_sequence.clear()
                key_presses.clear()
                sequence_counter += 1

if __name__ == "__main__":
    start_data_collection()
