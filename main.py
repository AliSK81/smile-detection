import cv2

from cnn_smile_detector import CNNSmileDetector
from genki4_dataset import Genki4Dataset
from live_prediction import LivePrediction

IMAGES_PATH = 'resources/genki4k/files'
LABELS_PATH = 'resources/genki4k/labels.txt'
UPDATED_IMAGES_PATH = 'resources/genki4k_updated/files'
UPDATED_LABELS_PATH = 'resources/genki4k_updated/labels.txt'
MODELS_PATH = 'resources/models'
VIDEO_PATH = 1  # webcam

IMAGE_MODE = cv2.IMREAD_GRAYSCALE
IMAGE_CHANNELS = 3 if IMAGE_MODE is cv2.IMREAD_COLOR else 1
IMAGE_SIZE = (128, 128)


def main():
    print('Load dataset..')
    dataset = Genki4Dataset(drop_bad_images=True,
                            resize_images=False,
                            crop_faces=False,
                            image_size=IMAGE_SIZE)

    dataset.load_dataset(images_path=UPDATED_IMAGES_PATH + 'clr',
                         labels_path=UPDATED_LABELS_PATH + 'clr',
                         image_mode=IMAGE_MODE)

    dataset.augment_data()
    #
    # print('Save dataset..')
    # dataset.save_dataset(images_path=UPDATED_IMAGES_PATH+'clr',
    #                      labels_path=UPDATED_LABELS_PATH+'clr')

    smile_detector = CNNSmileDetector(input_shape=(*IMAGE_SIZE, IMAGE_CHANNELS))

    print('Train model..')
    smile_detector.train_and_test(dataset.images, dataset.labels)

    print('Save model..')
    smile_detector.save_model(MODELS_PATH)

    print('Load model..')
    smile_detector.load_model(models_path=MODELS_PATH)

    print('Start prediction..')
    live_prediction = LivePrediction(smile_detector=smile_detector, image_size=IMAGE_SIZE, image_mode=IMAGE_MODE)
    live_prediction.run(source=VIDEO_PATH)


if __name__ == '__main__':
    main()
