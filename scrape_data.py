import cv2
import numpy as np
import time
from PIL import ImageGrab
from pynput import keyboard
import pygetwindow as gw
import os
import threading

# Folder to save data
data_folder = "ets2_data"
if not os.path.exists(data_folder):
    os.makedirs(data_folder)

# Global variables
screenshots = []
key_presses = []
capturing = False
last_saved_file_index = -1

# Function to capture the screen and apply pre-processing
def capture_screen(window_title=None):
    if window_title and capturing:
        game_window = gw.getWindowsWithTitle(window_title)[0]
        game_window.activate()
        x, y, width, height = game_window.left, game_window.top, game_window.width, game_window.height
        screen = np.array(ImageGrab.grab(bbox=(x, y, x + width, y + height)))
        screen = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)

        # Pre-processing: Resize and normalize
        screen = cv2.resize(screen, (224, 224))  # Example resize, adjust as needed
        screen = screen / 255.0  # Normalize pixel values

        return screen
    else:
        return None

# Function to log keyboard inputs
def on_press(key):
    global capturing, last_saved_file_index
    try:
        if key == keyboard.Key.page_up:
            capturing = True
            print("Data collection started.")
        elif key == keyboard.Key.page_down:
            capturing = False
            save_data()
            print("Data collection stopped.")
        elif capturing:
            print(f"Alphanumeric key pressed: {key.char}")
            key_presses.append((time.time(), key.char))
    except AttributeError:
        if capturing:
            print(f"Special key pressed: {key}")
            key_presses.append((time.time(), str(key)))

# Function to save the data
def save_data():
    global last_saved_file_index
    for i, (screenshot, key_press) in enumerate(zip(screenshots, key_presses)):
        index = last_saved_file_index + 1 + i
        # Save screenshot
        screenshot_path = os.path.join(data_folder, f"image_{index}.png")
        cv2.imwrite(screenshot_path, screenshot * 255)  # Convert back to original scale

        # Save key press
        with open(os.path.join(data_folder, "key_logs.txt"), "a") as file:
            file.write(f"{index}, {key_press[0]}, {key_press[1]}\n")
    last_saved_file_index += len(screenshots)
    screenshots.clear()
    key_presses.clear()

# Function to get the index of the last saved file
def get_last_saved_index():
    files = [f for f in os.listdir(data_folder) if f.endswith('.png')]
    if files:
        last_file = sorted(files)[-1]
        return int(last_file.split('_')[-1].split('.')[0])
    return -1

# Function to start data collection
def start_data_collection(window_title):
    listener = keyboard.Listener(on_press=on_press)
    listener.start()
    while True:
        if capturing:
            screen = capture_screen(window_title)
            if screen is not None:
                screenshots.append(screen)
        time.sleep(0.1)  # Adjust this based on capture frequency

if __name__ == "__main__":
    window_title = "Euro Truck Simulator 2"  # Adjust window title if needed
    last_saved_file_index = get_last_saved_index()
    start_data_collection(window_title)
