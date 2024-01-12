
# Autonomous Driving in Euro Truck Simulator 2

This project aims to demonstrate a simple form of autonomous driving in Euro Truck Simulator 2 (ETS2) using a Convolutional Neural Network (CNN) combined with Long Short-Term Memory (LSTM) networks. The model predicts driving actions based on sequences of screenshots from the game.

## Project Structure

- `data_collection.py`: Script to collect data from ETS2. It captures sequences of screenshots and corresponding keyboard inputs.
- `train_model.py`: Script to train the CNN-LSTM model on the collected data.
- `drive_autonomously.py`: Script to autonomously drive in ETS2 based on the trained model's predictions.

## Installation

1. **Clone the Repository**
   
   ```bash
   git clone https://github.com/your-repository/ets2-autonomous-driving.git
   cd ets2-autonomous-driving
   ```

2. **Install Required Libraries**
   
   You need Python 3.x and the following libraries: TensorFlow, NumPy, OpenCV, PyAutoGUI, and PyGetWindow.

   ```bash
   pip install tensorflow numpy opencv-python pyautogui pygetwindow
   ```

3. **Game Setup**
   
   Ensure Euro Truck Simulator 2 is installed and configured to run in a windowed mode with a consistent window title (e.g., "Euro Truck Simulator 2").

## Usage

1. **Data Collection**
   
   Run `data_collection.py` while playing ETS2 to collect training data.
   
   ```bash
   python data_collection.py
   ```
   
   Use Page Up to start collecting data and Page Down to stop. Data will be saved in the `ets2_data` directory.

2. **Training the Model**
   
   After collecting enough data, train the model using `train_model.py`.
   
   ```bash
   python train_model.py
   ```
   
   The trained model will be saved as `ets2_autonomous_driving_model.h5`.

3. **Autonomous Driving**
   
   Launch ETS2 and run `drive_autonomously.py` to start autonomous driving.
   
   ```bash
   python drive_autonomously.py
   ```

   The script will capture the game screen and use the model to predict and perform driving actions.

## Important Notes

- **Safety**: This project is intended for educational purposes and should be used responsibly.
- **Control**: Be prepared to take manual control of the game at any time.
- **Performance**: The accuracy and performance depend on the quality and quantity of training data.
- **Terms of Use**: Ensure compliance with the terms of use of ETS2.

## Contributions

Contributions to this project are welcome. Please submit a pull request or open an issue for bugs and feature requests.
