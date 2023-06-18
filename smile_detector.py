import pickle
from typing import List

import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split

from logger import Logger


class SmileDetector:
    def __init__(self, features: np.ndarray, labels: List[int]):
        self.features = features
        self.labels = labels
        self.model = svm.SVC(kernel='rbf')

    @Logger.log_time
    def train_and_test(self, test_size: float = 0.3) -> None:
        """Train and test the model."""
        x_train, x_test, y_train, y_test = \
            train_test_split(self.features, self.labels, test_size=test_size, random_state=42)
        self._train_model(x_train, y_train)
        accuracy = self._test_model(x_test, y_test)
        print(f'Test accuracy: {accuracy * 100:.2f}%')

    @Logger.log_time
    def _train_model(self, x_train: np.ndarray, y_train: List[int]) -> None:
        """Train the model."""
        self.model.fit(x_train, y_train)

    @Logger.log_time
    def _test_model(self, x_test: np.ndarray, y_test: List[int]) -> float:
        """Test the model."""
        return self.model.score(x_test, y_test)

    @Logger.log_time
    def save_model(self, filename: str) -> None:
        """Save the model to file."""
        with open(filename, 'wb') as f:
            pickle.dump(self.model, f)
