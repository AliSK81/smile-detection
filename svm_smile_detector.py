import pickle
from typing import List

import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split

from feature_extractor import FeatureExtractor
from logger import Logger


class SVMSmileDetector:
    def __init__(self):
        self.model = svm.SVC(kernel='rbf')
        self.feature_extractor = FeatureExtractor()

    @Logger.log_time
    def train_and_test(self, images: List[np.ndarray], labels: List[int], test_size: float = 0.3) -> None:
        """Train and test the model."""
        features = self.feature_extractor.extract_features(images)
        x_train, x_test, y_train, y_test = \
            train_test_split(features, labels, test_size=test_size, random_state=42)
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

    # @Logger.log_time
    def predict(self, image):
        features = self.feature_extractor.extract_features([image])
        return self.model.predict(features)

    @Logger.log_time
    def save_model(self, models_path: str) -> None:
        """Save the model to file."""
        with open(f'{models_path}/svm_model.pkl', 'wb') as f:
            pickle.dump(self.model, f)

    @Logger.log_time
    def load_model(self, models_path):
        """Load the model from file."""
        with open(f'{models_path}/svm_model.pkl', 'rb') as f:
            self.model = pickle.load(f)
