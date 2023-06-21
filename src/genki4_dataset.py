import os
from typing import Tuple, List

from face_detector import FaceDetector
from image_util import ImageUtil
from logger import Logger


class Genki4Dataset:
    def __init__(self, image_size=(128, 128),
                 drop_bad_images=True,
                 resize_images=True,
                 crop_faces=True):
        self.images = []
        self.labels = []
        self.face_detector = FaceDetector()
        self.image_size = image_size
        self.drop_bad_images = drop_bad_images
        self.resize_images = resize_images
        self.crop_faces = crop_faces

    @Logger.log_time
    def _load_labels(self, labels_path):
        """Load the labels from file."""
        with open(labels_path) as f:
            self.labels = [int(line.split()[0]) for line in f]

    @Logger.log_time
    def _load_images(self, images_path, mode):
        """Load the images from file."""
        for file in sorted(os.listdir(images_path)):
            img = ImageUtil.load_image(os.path.join(images_path, file), mode)

            if self.crop_faces:
                face_locations = self.face_detector.detect_faces(img)

                if face_locations:
                    img = ImageUtil.crop_faces(img, face_locations)[0]
                elif self.drop_bad_images:
                    self.labels.pop(len(self.images) - 1)
                    continue

            if self.resize_images:
                img = ImageUtil.resize_image(img, self.image_size)

            self.images.append(img)

    @Logger.log_time
    def load_dataset(self, images_path: str, labels_path: str, image_mode: int) -> Tuple[List, List]:
        """Load the dataset."""
        self._load_labels(labels_path)
        self._load_images(images_path, image_mode)
        return self.images, self.labels

    @Logger.log_time
    def save_dataset(self, images_path: str, labels_path: str):
        """Save the dataset to file."""
        if not os.path.exists(images_path):
            os.makedirs(images_path)

        for i, image in enumerate(self.images):
            file_path = os.path.join(images_path, f"{i:04d}.jpg")
            ImageUtil.save_image(file_path, image)

        with open(labels_path, "w") as f:
            for label in self.labels:
                f.write(f"{label}\n")
