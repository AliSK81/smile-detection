from typing import Tuple, List

import cv2
import numpy as np


class ImageUtil:
    @staticmethod
    def load_image(file_path: str, mode: int = cv2.IMREAD_GRAYSCALE) -> np.ndarray:
        """Load an image from file path."""
        return cv2.imread(file_path, mode)

    @staticmethod
    def save_image(file_path: str, image) -> np.ndarray:
        """Load an image from file path."""
        return cv2.imwrite(file_path, image)

    @staticmethod
    def resize_image(image: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
        """Resize an image."""
        return cv2.resize(image, size)

    @staticmethod
    def crop_faces(image: np.ndarray, face_locations) -> List[np.ndarray]:
        """Crop faces from an image."""
        cropped_faces = [image[top:bottom, left:right] for (top, right, bottom, left) in face_locations]
        return cropped_faces

    @staticmethod
    def convert_to_gray(image: np.ndarray) -> np.ndarray:
        """Convert an image to grayscale."""
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
