from typing import List

import cv2
import numpy as np


class DataAugmenter:
    def __init__(self, rotation_range: float = 20, width_shift_range: float = 0.2,
                 height_shift_range: float = 0.2, zoom_range: float = 0.2,
                 horizontal_flip: bool = True):
        self.rotation_range = rotation_range
        self.width_shift_range = width_shift_range
        self.height_shift_range = height_shift_range
        self.zoom_range = zoom_range
        self.horizontal_flip = horizontal_flip

    def random_rotate(self, image):
        angle = np.random.uniform(-self.rotation_range, self.rotation_range)
        rows, cols = image.shape[:2]
        rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
        rotated_image = cv2.warpAffine(image, rotation_matrix, (cols, rows), borderMode=cv2.BORDER_REPLICATE)
        return rotated_image

    def random_shift(self, image):
        w_shift = np.random.uniform(-self.width_shift_range, self.width_shift_range) * image.shape[1]
        h_shift = np.random.uniform(-self.height_shift_range, self.height_shift_range) * image.shape[0]
        translation_matrix = np.float32([[1, 0, w_shift], [0, 1, h_shift]])
        shifted_image = cv2.warpAffine(image, translation_matrix, (image.shape[1], image.shape[0]),
                                       borderMode=cv2.BORDER_REPLICATE)
        return shifted_image

    def random_horizontal_flip(self, image):
        if np.random.random() < 0.5 and self.horizontal_flip:
            flipped_image = cv2.flip(image, 1)
            return flipped_image
        else:
            return image

    def augment(self, images: List[np.ndarray], labels: [int], output_size: int = 1):
        """
        Augments the input images and returns the augmented images.
        """
        augmented_images = []
        augmented_labels = labels * output_size

        for _ in range(output_size):
            for image, label in zip(images, labels):
                image = self.random_rotate(image)
                # image = self.random_shift(image)
                # image = self.random_horizontal_flip(image)
                augmented_images.append(image)

        return augmented_images, augmented_labels


if __name__ == '__main__':
    image = cv2.imread('resources/genki4k_updated/filesclr/0000.jpg')
    da = DataAugmenter()

    print(image.shape)
    im, la = da.augment([image], [1], 2)
    cv2.imshow('res', im[0])
    cv2.waitKey(0)
    cv2.imshow('res', im[1])
    cv2.waitKey(0)
