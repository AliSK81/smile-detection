import pickle
from typing import List

import numpy as np
from sklearn import svm

from feature_extractor import FeatureExtractor
from logger import Logger


class SVMSmileDetector:
    def __init__(self):
        self.model = svm.SVC(kernel='rbf')
        self.feature_extractor = FeatureExtractor()

    @Logger.log_time
    def train(self, x_train: List[np.ndarray], x_test: List[int]) -> None:
        """Train the model."""
        x_train = self.feature_extractor.extract_features(x_train)
        y_train = x_test
        self.model.fit(x_train, y_train)

    @Logger.log_time
    def test(self, y_train: List[np.ndarray], y_test: List[int]) -> float:
        """Test the model."""
        x_test = self.feature_extractor.extract_features(y_train)
        y_test = y_test
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
