# Task 1: Data Simulation:

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.utils import to_categorical
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import time


def generate_sequential_data(num_sequences, sequence_length, num_features):
    # Generate random sequential data
    data = np.random.rand(num_sequences, sequence_length, num_features)
    return data

def add_failure_scenarios(data, failure_prob=0.1):
    # Simulate failure scenarios
    num_sequences = data.shape[0]
    num_failures = int(num_sequences * failure_prob)

    # Injecting failure data randomly
    failure_indices = np.random.choice(num_sequences, num_failures, replace=False)
    for idx in failure_indices:
        # Simulate a failure scenario (modifying data points)
        data[idx, :, :] = np.random.rand(data.shape[1], data.shape[2])  # Replace with failure logic

    return data

# Generate data
num_sequences = 100
sequence_length = 50
num_features = 3

# Generate sequential data
data = generate_sequential_data(num_sequences, sequence_length, num_features)

# Add failure scenarios
data_with_failures = add_failure_scenarios(data, failure_prob=0.1)

print(data_with_failures.shape)

# Task 2: Data Preprocessing

# Preprocess the generated data
def preprocess_sequential_data(data):
    # Reshape the data for preprocessing
    num_sequences, sequence_length, num_features = data.shape
    data = data.reshape(-1, num_features)

    # Standardize the data
    scaler = StandardScaler()
    data = scaler.fit_transform(data)

    # Reshape the data back to the original shape
    data = data.reshape(num_sequences, sequence_length, num_features)

    return data

# Split data into training and testing sets
def split_data(data, test_size=0.2):
    X = data  # Features
    y = np.zeros(X.shape[0])  # Labels (0 for normal, 1 for failure) - we can adjust this based on our data

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    return X_train, X_test, y_train, y_test

# Preprocess the generated data
preprocessed_data = preprocess_sequential_data(data_with_failures)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = split_data(preprocessed_data, test_size=0.2)

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

# Task 3: LSTM Model

# Create an LSTM model
def create_lstm_model(input_shape, num_classes):
    model = keras.Sequential()

    # LSTM layer with 50 units
    model.add(layers.LSTM(50, input_shape=input_shape, activation='relu', return_sequences=True))

    # Output layer with a single neuron for binary classification
    model.add(layers.Dense(1, activation='sigmoid'))

    # Compile the model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

# Input shape based on the training data
input_shape = (X_train.shape[1], X_train.shape[2])
num_classes = 1  # Binary classification (0 for normal, 1 for failure)

# Create the LSTM model
lstm_model = create_lstm_model(input_shape, num_classes)

# Display the model summary
lstm_model.summary()

# Task 4: Model Training

# Can replace this with actual training data and labels
X_train = np.random.rand(100, 50, 3)
y_train = np.random.randint(2, size=100)  # Assuming binary labels (0 or 1)

# One-hot encode the labels
y_train_encoded = to_categorical(y_train, num_classes=2)

# Create an LSTM model
def create_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, input_shape=input_shape))
    model.add(Dense(2, activation='softmax'))  # 2 classes (0 and 1)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Set the number of training epochs and batch size
epochs = 10
batch_size = 32

# Create the LSTM model
input_shape = (X_train.shape[1], X_train.shape[2])
lstm_model = create_lstm_model(input_shape)

# Train the LSTM model using the training data
history = lstm_model.fit(X_train, y_train_encoded, epochs=epochs, batch_size=batch_size, verbose=1)

# Monitor the training process
print("Training completed!")

# Generate predictions on the training data
y_pred_encoded = lstm_model.predict(X_train)

# Convert predictions back to labels
y_pred = np.argmax(y_pred_encoded, axis=1)

# Evaluate the model
accuracy = accuracy_score(y_train, y_pred)
confusion = confusion_matrix(y_train, y_pred)
report = classification_report(y_train, y_pred)

print(f"Accuracy: {accuracy}")
print(f"Confusion Matrix:\n{confusion}")
print(f"Classification Report:\n{report}")

lstm_model.save("12200276_new.h5")

# Task 5: Real-time Simulation

# Function to simulate real-time data
def simulate_real_time(lstm_model, initial_data, num_steps):
    current_data = initial_data.copy()

    for step in range(num_steps):
        # Replace this with actual sensor data retrieval logic
        new_data_point = np.random.rand(1, initial_data.shape[1])

        # Add the new data point to the current data
        current_data = np.concatenate([current_data, new_data_point], axis=0)

        # Maintain a rolling window of data (sequence length)
        current_data = current_data[-sequence_length:]

        # Predict using the LSTM model
        prediction = lstm_model.predict(current_data.reshape(1, sequence_length, num_features))

        # Check if a failure is predicted (can define a threshold)
        if prediction[0, 1] > 0.5:
            print(f"Failure predicted at step {step}!")


        # Introduce a delay to simulate real-time
        time.sleep(1)  # Simulated 1 second interval between data points

# Can replace these with actual values
sequence_length = 50  # Should match your training sequence length
num_features = 3  # Should match the number of features in your data

# Example usage:
# Replace with the path to trained LSTM model
model_path = "12200276_new.h5"
lstm_model = keras.models.load_model(model_path)

# Can replace this with actual initial data (shape: (sequence_length, num_features))
initial_data = np.random.rand(sequence_length, num_features)
num_steps = 10  # Define the number of steps to simulate

# Simulate real-time data
simulate_real_time(lstm_model, initial_data, num_steps)

# Task 6: Complete the Assignment

# Define the number of iterations to run the simulation
num_iterations = 2  # Can be changed this to the desired number of iterations

# Loop through the specified number of iterations
for iteration in range(num_iterations):
    print(f"Iteration {iteration + 1}/{num_iterations}:")

    # Simulate real-time data for each iteration
    simulate_real_time(lstm_model, initial_data, num_steps)


print("Simulation completed for all iterations.")

# The additional challenge

# Function to visualize real-time sensor data and predictions

def visualize_real_time(sensor_data, predicted_data):
    plt.ion()  # Turn on interactive mode for real-time plotting
    num_steps = len(sensor_data)

    for step in range(num_steps):
        plt.clf()  # Clear the previous plot
        plt.subplot(2, 1, 1)
        plt.plot(sensor_data[:step + 1, 0], label='Temperature')
        plt.plot(sensor_data[:step + 1, 1], label='Vibration')
        plt.xlabel('Time Steps')
        plt.ylabel('Sensor Data')
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.plot(predicted_data[:step + 1], label='Predicted Data', color='r')
        plt.xlabel('Time Steps')
        plt.ylabel('Predicted Data')
        plt.legend()

        plt.draw()
        plt.pause(0.1)
        time.sleep(0.1)

    plt.ioff()  # Turn off interactive mode after visualization

# Example usage:
# Provide your sensor data and predicted data
sensor_data = np.random.rand(5, 2)  # Can be replaced with actual sensor data
predicted_data = np.random.rand(2)  # Can be replaced with actual predicted data

visualize_real_time(sensor_data, predicted_data)
