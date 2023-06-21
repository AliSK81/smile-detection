from typing import List

import numpy as np
import tensorflow as tf
from keras.saving.saving_api import load_model
from tensorflow.python.keras import models, layers

from logger import Logger


class CNNSmileDetector:
    def __init__(self, input_shape):
        self.model = self.create_model(input_shape)

    def create_model(self, input_shape):
        model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
            tf.keras.layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.BatchNormalization(),
            layers.AveragePooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam',
                           loss='binary_crossentropy',
                           metrics=['accuracy'])
        return model

    @Logger.log_time
    def train(self, x_train: List[np.ndarray], y_train: List[int], x_val: List[np.ndarray], y_val: List[int]) -> None:
        """Train the model."""
        if len(x_train[0].shape) == 2:
            x_train = np.expand_dims(x_train, axis=-1) / 255.0
            x_val = np.expand_dims(x_val, axis=-1) / 255.0

        self.model.fit(tf.convert_to_tensor(x_train),
                       tf.convert_to_tensor(y_train),
                       batch_size=30,
                       epochs=10,
                       validation_data=(tf.convert_to_tensor(x_val), tf.convert_to_tensor(y_val)))

    @Logger.log_time
    def test(self, x_test: List[np.ndarray], y_test: List[int]) -> float:
        """Test the model."""
        if len(x_test[0].shape) == 2:
            x_test = np.expand_dims(x_test, axis=-1) / 255.0

        loss, accuracy = self.model.evaluate(tf.convert_to_tensor(x_test),
                                             tf.convert_to_tensor(y_test))
        return accuracy

    def predict(self, image: np.ndarray):
        """Predict the output for a given image."""
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=-1) / 255.0
        image = np.expand_dims(image, axis=0)
        image = tf.convert_to_tensor(image)
        output = self.model.predict(image)
        smile_probability = output[0][0]
        return [1] if smile_probability >= 0.5 else [0]

    @Logger.log_time
    def save_model(self, models_path: str) -> None:
        """Save the model to file."""
        self.model.save(f'{models_path}/cnn_model')

    @Logger.log_time
    def load_model(self, models_path):
        """Load the model from file."""
        self.model = load_model(f'{models_path}/cnn_model')
