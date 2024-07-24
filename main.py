import pickle
import tensorflow as tf
from tensorflow import keras
import numpy as np

# Function to convert a scikit-learn MLP model to a Keras model
def convert_sklearn_to_keras(sklearn_model):
    model = keras.Sequential()
    for units in sklearn_model.hidden_layer_sizes:
        model.add(keras.layers.Dense(units, activation='relu'))
    model.add(keras.layers.Dense(sklearn_model.n_outputs_, activation='softmax'))
    return model

# Load the scikit-learn model from the .pkl file
with open('model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

# Extract the actual model if it's in a pipeline (adjust 'classifier' if necessary)
if hasattr(loaded_model, 'named_steps'):
    sklearn_model = loaded_model.named_steps['classifier']
else:
    sklearn_model = loaded_model

# Ensure the model is an MLP and convert to Keras
if hasattr(sklearn_model, 'hidden_layer_sizes'):
    keras_model = convert_sklearn_to_keras(sklearn_model)

    # Compile the Keras model (use dummy values for loss and optimizer since we won't train it)
    keras_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Convert the Keras model to a TensorFlow Lite model
    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    tflite_model = converter.convert()

    # Save the TensorFlow Lite model to a file
    with open('model.tflite', 'wb') as file:
        file.write(tflite_model)

    print("Conversion completed: model.tflite has been saved.")
else:
    print("The loaded model is not an MLP and cannot be converted using this script.")
