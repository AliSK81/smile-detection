from typing import Tuple, List

import cv2
import face_recognition
import numpy as np

from logger import Logger


class FaceDetector:
    def __init__(self, model: str = 'fog'):
        self.model = model

    @Logger.log_time
    def detect_faces(self, image: np.ndarray, method: str = 'fr') -> List[Tuple[int, int, int, int]]:
        if method == 'fr':
            return self._detect_faces_fr(image)
        if method == 'cv2':
            return self._detect_faces_cv2(image)
        raise Exception(f'invalid method: {method}')

    @Logger.log_time
    def _detect_faces_fr(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect faces in an image using face_recognition."""
        face_locations = face_recognition.face_locations(image, model=self.model)
        return face_locations

    @Logger.log_time
    def _detect_faces_cv2(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect faces in an image using OpenCV."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier('resources/haarcascade_frontalface_default.xml')
        face_locations = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        face_locations = [(y, x + w, y + h, x) for (x, y, w, h) in face_locations]
        return face_locations
