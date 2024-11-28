# Importing the required Python libraries similar to labs
import numpy as np
import os
import tensorflow as tf
# import torch
import matplotlib.pyplot as plt
import joblib

# Scikit Libraries
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from skimage.io import imread
from skimage.transform import resize

# print(torch.version.cuda)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Dataset folder is where the data is stored.
# https://www.kaggle.com/datasets/flo2607/traffic-signs-classification/data
datasetFolder = r'C:\Users\lojan\OneDrive\Documents\COMPSYS306\archive\myData'
data_file = 'dataset.joblib'
model_file = 'mlp_model.joblib'  # MLP model
test_data_file = 'test_data.joblib'  # Test data
history_file = 'training_history.joblib' # Training History
# Getting the folders inside the dataset folder using os.listdir
# https://stackoverflow.com/a/3207973
signCategories = os.listdir(datasetFolder)
num_classes = len(signCategories)
# Printing it out to confirm that the folders show up and correct amount
print(signCategories, "\nAmount of folders in datasetFolder:", num_classes)


def open_dataset(dataset_folder, sign_categories):
    # Initializing empty lists to store flattened pixel data from the images and target labels
    flattened_data = []
    target_labels = []

    # Image processing loop for signCategories
    for i in sign_categories:
        # Print category number
        print(f'Loading category {i}')
        image_path = os.path.join(dataset_folder, i)

        for j in os.listdir(image_path):
            img_p = os.path.join(image_path, j)
            img_arr = imread(img_p)

            # If image has more than 3 channels, skip
            if img_arr.ndim > 3:
                print(f"Image with extra channel: {image_path}")
                continue

            # Without using .astype(np.float16), the dataset file size is greater than 1gb
            img_resized = resize(img_arr, (28, 28, 3)).astype(np.float16)
            # Normalizing the image
            img_normalized = img_resized / 255.0
            # Add the normalized image to the flattened_data list after flattening
            flattened_data.append(img_normalized)
            # Add the index i of signCategories to target_labels list
            target_labels.append(signCategories.index(i))
        print(f'Loaded category: {i}')
    # Convert flattened_data and target_labels lists to numpy arrays
    flat_data_np = np.array(flattened_data)
    target_np = np.array(target_labels)
    return flat_data_np, target_np


# Load or process dataset
if os.path.exists(data_file):
    img, target = joblib.load(data_file)
else:
    img, target = open_dataset(datasetFolder, signCategories)
    joblib.dump((img, target), data_file)

# Train/Test split
train_data, test_data, train_target, test_target = train_test_split(img, target, test_size=0.2, random_state=77)

# Save the test dataset for later use
joblib.dump((test_data, test_target), test_data_file)

# Define and save MLP model using joblib
def save_mlp_as_joblib(model_file, history_file):
    # Defining MLP model structure
    mlpModel = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28, 3)),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(num_classes)  # Set up for 43 different classes
    ])
    mlpModel.compile(optimizer='adam',
                     loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                     metrics=['accuracy'])

    # Train the model and capture the history
    history = mlpModel.fit(train_data, train_target, epochs=30, validation_data=(test_data, test_target))

    # Save the model architecture and weights using joblib
    model_dict = {
        "architecture": mlpModel.to_json(),  # Model architecture in JSON
        "weights": mlpModel.get_weights()  # Model weights
    }
    joblib.dump(model_dict, model_file)

    # Save the training history
    joblib.dump(history.history, history_file)

# Save the model and history if not already saved
if not os.path.exists(model_file):
    save_mlp_as_joblib(model_file, history_file)
else:
    print(f"MLP model already saved at {model_file}")

# Load MLP model from joblib
def load_mlp_from_joblib(model_file):
    model_dict = joblib.load(model_file)
    mlpModel = tf.keras.models.model_from_json(model_dict["architecture"])
    mlpModel.set_weights(model_dict["weights"])
    mlpModel.compile(optimizer='adam',
                     loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                     metrics=['accuracy'])
    return mlpModel

# Load and test the model
mlpModel = load_mlp_from_joblib(model_file)
test_loss, test_accuracy = mlpModel.evaluate(test_data, test_target)
print(f'Test Accuracy: {test_accuracy}')

# Generate the classification report
predictions = mlpModel.predict(test_data)
predictions = np.argmax(predictions, axis=1)
print(classification_report(test_target, predictions))

# Plot the training history if it exists
if os.path.exists(history_file):
    history = joblib.load(history_file)
    epochs = range(1, len(history['accuracy']) + 1)

    # Plot accuracy and loss graphs
    plt.figure(figsize=(12, 5))

    # Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['accuracy'], label='Train Accuracy')
    plt.plot(epochs, history['val_accuracy'], label='Test Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Epoch vs Accuracy')
    plt.legend()

    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['loss'], label='Train Loss')
    plt.plot(epochs, history['val_loss'], label='Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Epoch vs Loss')
    plt.legend()

    plt.show()