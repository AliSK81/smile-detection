from typing import List

import cv2
import numpy as np


class DataAugmenter:
    def __init__(self, rotation_range: float = 20, ):
        self.rotation_range = rotation_range

    def random_rotate(self, image):
        angle = np.random.uniform(-self.rotation_range, self.rotation_range)
        rows, cols = image.shape[:2]
        rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
        rotated_image = cv2.warpAffine(image, rotation_matrix, (cols, rows), borderMode=cv2.BORDER_REPLICATE)
        return rotated_image

    def augment(self, images: List[np.ndarray], labels: [int], output_size: int = 1):
        """Augments the input images."""
        augmented_images = []
        augmented_labels = labels * output_size

        for _ in range(output_size):
            for image, label in zip(images, labels):
                image = self.random_rotate(image)
                augmented_images.append(image)

        return augmented_images, augmented_labels


if __name__ == '__main__':
    image = cv2.imread('../resources/genki4k_updated/filesclr/0000.jpg')
    da = DataAugmenter()

    print(image.shape)
    im, la = da.augment([image], [1], 2)
    cv2.imshow('res', im[0])
    cv2.waitKey(0)
    cv2.imshow('res', im[1])
    cv2.waitKey(0)
