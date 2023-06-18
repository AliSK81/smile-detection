from typing import List

import numpy as np
from skimage import feature

from logger import Logger


class FeatureExtractor:
    def __init__(self):
        pass

    @Logger.log_time
    def extract_features(self, images: List[np.ndarray]) -> np.ndarray:
        """Extract features from the images."""
        hog_features = self._extract_hog_features(images)
        lbp_features = self._extract_lbp_features(images)
        features = np.concatenate((hog_features, lbp_features), axis=1)
        return features

    @Logger.log_time
    def _extract_hog_features(self, images: List[np.ndarray]) -> List[np.ndarray]:
        """Extract HOG features from the images."""
        hog_features = [feature.hog(img, orientations=20, pixels_per_cell=(16, 16), cells_per_block=(2, 2))
                        for img in images]
        return hog_features

    @Logger.log_time
    def _extract_lbp_features(self, images: List[np.ndarray]) -> List[np.ndarray]:
        """Extract LBP features from the images."""
        lbp_features = [feature.local_binary_pattern(img, P=8, R=1, method='uniform').flatten()
                        for img in images]
        histograms = [np.histogram(features, bins=10)[0] for features in lbp_features]
        lbp_histograms = [histogram / np.sum(histogram) for histogram in histograms]
        return lbp_histograms
