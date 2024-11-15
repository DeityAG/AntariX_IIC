# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sgp4.earth_gravity import wgs72
from sgp4.io import twoline2rv
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Function to parse TLE data from the given format
def read_tle_file(file_path):
    satellites = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for i in range(0, len(lines), 3):  # Each satellite has 3 lines: name + 2 TLE lines
            satellite = {
                "name": lines[i].strip(),
                "line1": lines[i + 1].strip(),
                "line2": lines[i + 2].strip()
            }
            satellites.append(satellite)
    return pd.DataFrame(satellites)

# Load TLE data from file
file_path = 'stations.txt'  # Replace with your actual file path
satellite_data = read_tle_file(file_path)
print(satellite_data.head())

def compute_original_trajectory(satellite_data, satellite_name, days=30):
    # Get the TLE data for the specified satellite
    satellite_row = satellite_data[satellite_data['name'] == satellite_name].iloc[0]
    satellite = twoline2rv(satellite_row['line1'], satellite_row['line2'], wgs72)

    # Calculate the original position over the specified period
    original_positions = []
    current_time = datetime.utcnow()
    for i in range(days):
        prediction_time = current_time + timedelta(days=i)
        position, _ = satellite.propagate(
            prediction_time.year,
            prediction_time.month,
            prediction_time.day,
            prediction_time.hour,
            prediction_time.minute,
            prediction_time.second
        )
        original_positions.append(np.linalg.norm(position))  # Taking norm to get scalar distance

    return np.array(original_positions)

# Compute original trajectory for "ISS (ZARYA)" over 30 days
original_positions = compute_original_trajectory(satellite_data, "ISS (ZARYA)", days=30)
print("Original Trajectory:", original_positions)

from sgp4.earth_gravity import wgs72
from sgp4.io import twoline2rv
from datetime import datetime

def predict_positions(satellite_data, prediction_time):
    results = []
    for sat in satellite_data:
        satellite = twoline2rv(sat['line1'], sat['line2'], wgs72)
        position, velocity = satellite.propagate(
            prediction_time.year,
            prediction_time.month,
            prediction_time.day,
            prediction_time.hour,
            prediction_time.minute,
            prediction_time.second
        )
        results.append({
            "name": sat['name'],
            "position": position,
            "velocity": velocity
        })
    return results

def sgp4_predict(tle_line1, tle_line2, prediction_time):
    satellite = twoline2rv(tle_line1, tle_line2, wgs72)
    position, velocity = satellite.propagate(
        prediction_time.year,
        prediction_time.month,
        prediction_time.day,
        prediction_time.hour,
        prediction_time.minute,
        prediction_time.second
    )
    return np.array(position)

# Example prediction
prediction_time = datetime.utcnow()
sample_tle = satellite_data.iloc[0]
predicted_position = sgp4_predict(sample_tle['line1'], sample_tle['line2'], prediction_time)
print(f"Predicted Position: {predicted_position}")

def create_features(tle_data):
    features = []
    for _, tle in tle_data.iterrows():
        # Create Satellite object
        satellite = twoline2rv(tle['line1'], tle['line2'], wgs72)

        # Append features extracted from Satellite object
        features.append([
            satellite.no_kozai,   # Mean motion (rad/min)
            satellite.ecco,       # Eccentricity
            satellite.inclo,      # Inclination (rad)
            satellite.nodeo,      # Right ascension of ascending node (rad)
            satellite.argpo,      # Argument of perigee (rad)
            satellite.mo,         # Mean anomaly (rad)
            satellite.bstar       # Drag term
        ])
    return np.array(features)

# Generate features
features = create_features(satellite_data)
print(f"Generated Features: {features}")

class HybridOrbitPredictor:
    def __init__(self):
        self.ml_model = Sequential([
            LSTM(64, return_sequences=True, input_shape=(1, 7)),
            LSTM(32),
            Dense(32, activation='relu'),
            Dense(3)  # Output: position corrections (x, y, z)
        ])
        self.scaler = StandardScaler()

    def train(self, X_train, y_train, epochs=100):
        # Fit and transform the training data
        self.scaler.fit(X_train)
        X_train = self.scaler.transform(X_train)
        X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))  # Reshape for LSTM

        # Compile and train the model
        self.ml_model.compile(optimizer='adam', loss='mse')
        self.ml_model.fit(X_train, y_train, epochs=epochs, verbose=1)

    def predict(self, tle_line1, tle_line2, prediction_time):
        # Get SGP4 position
        sgp4_position = sgp4_predict(tle_line1, tle_line2, prediction_time)

        # Create features and scale them
        features = create_features(pd.DataFrame([{"line1": tle_line1, "line2": tle_line2}]))
        scaled_features = self.scaler.transform(features)
        scaled_features = scaled_features.reshape((1, 1, -1))  # Reshape for LSTM

        # Get correction from the ML model
        correction = self.ml_model.predict(scaled_features)

        # Return the corrected position
        return sgp4_position + correction[0]

from datetime import datetime, timedelta
import numpy as np

def evaluate_model(tle_data, prediction_days=[10, 20, 30]):
    results = {}
    model = HybridOrbitPredictor()

    # Generate features and labels for training
    X = create_features(tle_data)
    y = np.random.random((len(X), 3))  # Placeholder for true correction data

    # Split data for training and fit the model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model.train(X_train, y_train)

    for index, satellite in tle_data.iterrows():
        satellite_name = satellite['name']
        print(f"Evaluating for satellite: {satellite_name}")  # Debugging line

        for days in prediction_days:
            predictions = []
            current_time = datetime.utcnow()

            # Generate predictions for each day in the prediction period
            for i in range(days):
                pred_time = current_time + timedelta(days=i)
                pred_position = model.predict(satellite['line1'], satellite['line2'], pred_time)
                predictions.append(pred_position)

            # Placeholder for true positions to compute errors
            true_positions = np.random.random((days, 3)) * 7000  # Replace with actual data if available
            errors = np.linalg.norm(np.array(predictions) - np.array(true_positions), axis=1)

            # Store results with key format "satellite_name_daysdays"
            key = f"{satellite_name}_{days}days"
            print(f"Storing results for {key}: Mean error = {np.mean(errors)}, Max error = {np.max(errors)}")  # Debugging line
            results[key] = {
                'mean_error': np.mean(errors),
                'max_error': np.max(errors),
                'errors': errors
            }

    return results

def plot_trajectory_with_prediction(results, satellite_name, original_positions, days=30):
    plt.figure(figsize=(10, 6))

    # Plot original trajectory (in green)
    plt.plot(
        range(days),
        original_positions[:days],
        label="Original Trajectory",
        color="green"
    )

    # Plot predicted trajectory (in red)
    key = f"{satellite_name}_{days}days"
    if key in results:
        predicted_positions = original_positions[:days] + results[key]['errors']  # Predicted by adding forecasted errors
        plt.plot(
            range(days),
            predicted_positions,
            label="Predicted Trajectory",
            color="red"
        )
    else:
        print(f"Missing data for {key}")

    # Add labels, title, legend, and grid
    plt.title(f"Trajectory Prediction for {satellite_name}")
    plt.xlabel("Days")
    plt.ylabel("Position (km)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Plot the results for a 30-day period
plot_trajectory_with_prediction(results, "ISS (ZARYA)", original_positions, days=30)

import matplotlib.pyplot as plt

def plot_single_satellite(results, satellite_name):
    plt.figure(figsize=(10, 6))

    # Plot errors for different prediction periods (10, 20, 30 days)
    for days in [10, 20, 30]:
        key = f"{satellite_name}_{days}days"
        if key in results:
            print(f"Plotting data for {key}: {results[key]['errors']}")  # Debugging line
            plt.plot(
                range(days),
                results[key]['errors'],
                label=f"{days} days"
            )
        else:
            print(f"Missing data for {key}")  # Debugging line

    # Add labels, title, legend, and grid
    plt.title(f"Prediction Errors for {satellite_name}")
    plt.xlabel("Days")
    plt.ylabel("Position Error (km)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Specify the name of the satellite you want to visualize
satellite_name = "ISS (ZARYA)"

# Plot results for the chosen satellite
plot_single_satellite(results, satellite_name)

# Read TLE data from the file
file_path = 'stations.txt'  # Replace with your actual file path
satellite_data = read_tle_file(file_path)
print(satellite_data.head())

from sgp4.io import twoline2rv
from sgp4.earth_gravity import wgs72
from datetime import datetime, timedelta

def compute_original_trajectory(satellite_data, satellite_name, days=30):
    satellite_row = satellite_data[satellite_data['name'] == satellite_name].iloc[0]
    satellite = twoline2rv(satellite_row['line1'], satellite_row['line2'], wgs72)

    # Calculate the original position over a specified period
    original_positions = []
    current_time = datetime.utcnow()
    for i in range(days):
        prediction_time = current_time + timedelta(days=i)
        position, _ = satellite.propagate(
            prediction_time.year,
            prediction_time.month,
            prediction_time.day,
            prediction_time.hour,
            prediction_time.minute,
            prediction_time.second
        )
        original_positions.append(np.linalg.norm(position))  # Taking norm to get scalar distance

    return original_positions

# Example: Compute original trajectory for "ISS (ZARYA)" over 30 days
original_positions = compute_original_trajectory(satellite_data, "ISS (ZARYA)", days=30)
print("Original Trajectory:", original_positions)

def plot_single_satellite_with_original(results, satellite_name, original_positions):
    plt.figure(figsize=(10, 6))

    # Plot original trajectory
    plt.plot(
        range(len(original_positions)),
        original_positions,
        label="Original Trajectory",
        linestyle="--",
        color="black"
    )

    # Plot errors for different prediction periods (10, 20, 30 days)
    for days in [10, 20, 30]:
        key = f"{satellite_name}_{days}days"
        if key in results:
            predicted_positions = original_positions[:days] + results[key]['errors']
            plt.plot(
                range(days),
                predicted_positions,
                label=f"{days} days forecast"
            )

    # Add labels, title, legend, and grid
    plt.title(f"Prediction Errors for {satellite_name} with Original Trajectory")
    plt.xlabel("Days")
    plt.ylabel("Position (km)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Plot the results
plot_single_satellite_with_original(results, "ISS (ZARYA)", original_positions)



