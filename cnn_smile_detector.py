from typing import List

import numpy as np
import tensorflow as tf
from keras.saving.saving_api import load_model
from sklearn.model_selection import train_test_split
from tensorflow.python.keras import models, layers

from logger import Logger


class CNNSmileDetector:
    def __init__(self, input_shape):
        self.model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ])

    @Logger.log_time
    def train_and_test(self, images: List[np.ndarray], labels: List[int], test_size: float = 0.3) -> None:
        """Train and test the model."""
        x_train, x_test, y_train, y_test = \
            train_test_split(images, labels, test_size=test_size, random_state=42)

        if len(images[0].shape) == 2:
            x_train = np.expand_dims(x_train, axis=-1) / 255.0
            x_test = np.expand_dims(x_test, axis=-1) / 255.0

        self._train_model(x_train, y_train)
        accuracy = self._test_model(x_test, y_test)
        print(f'Test accuracy: {accuracy * 100:.2f}%')

    @Logger.log_time
    def _train_model(self, x_train: List[np.ndarray], y_train: List[int]) -> None:
        """Train the model."""
        x_train = tf.convert_to_tensor(x_train)
        y_train = tf.convert_to_tensor(y_train)
        self.model.compile(optimizer='adam',
                           loss='binary_crossentropy',
                           metrics=['accuracy'])

        self.model.fit(x_train, y_train, batch_size=20, epochs=15)

    @Logger.log_time
    def _test_model(self, x_test: List[np.ndarray], y_test: List[int]) -> float:
        """Test the model."""
        x_test = tf.convert_to_tensor(x_test)
        y_test = tf.convert_to_tensor(y_test)
        loss, accuracy = self.model.evaluate(x_test, y_test)
        return accuracy

    # @Logger.log_time
    def predict(self, image: np.ndarray):
        """Predict the output for a given image."""
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=-1) / 255.0
        image = np.expand_dims(image, axis=0)
        image = tf.convert_to_tensor(image)
        output = self.model.predict(image)
        smile_probability = output[0][0]
        print(smile_probability)
        return [1] if smile_probability >= 0.5 else [0]
        # return output[0][0]

    @Logger.log_time
    def save_model(self, models_path: str) -> None:
        """Save the model to file."""
        self.model.save(f'{models_path}/cnn_model')

    @Logger.log_time
    def load_model(self, models_path):
        """Load the model from file."""
        self.model = load_model(f'{models_path}/cnn_model')
