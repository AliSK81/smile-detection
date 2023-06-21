import cv2

from cnn_smile_detector import CNNSmileDetector
from data_agumenter import DataAugmenter
from data_splitter import DataSplitter
from genki4_dataset import Genki4Dataset
from live_prediction import LivePrediction

IMAGES_PATH = '../resources/genki4k/files'
LABELS_PATH = '../resources/genki4k/labels.txt'
UPDATED_IMAGES_PATH = '../resources/genki4k_updated/files'
UPDATED_LABELS_PATH = '../resources/genki4k_updated/labels.txt'
MODELS_PATH = '../resources/models'
VIDEO_PATH = 0  # webcam

IMAGE_MODE = cv2.IMREAD_GRAYSCALE
IMAGE_CHANNELS = 3 if IMAGE_MODE is cv2.IMREAD_COLOR else 1
IMAGE_SIZE = (128, 128)


def main():
    print('Load dataset..')
    dataset = Genki4Dataset(drop_bad_images=True,
                            resize_images=False,
                            crop_faces=False,
                            image_size=IMAGE_SIZE)

    dataset.load_dataset(images_path=UPDATED_IMAGES_PATH,
                         labels_path=UPDATED_LABELS_PATH,
                         image_mode=IMAGE_MODE)

    print('Split data')
    data_splitter = DataSplitter(train_size=0.6, val_size=0.1, test_size=0.3)

    x_train, y_train, x_val, y_val, x_test, y_test = data_splitter.split(data=dataset.images, labels=dataset.labels)

    print('Augment data')
    data_augmenter = DataAugmenter()
    augmented_data = data_augmenter.augment(images=x_train, labels=y_train, output_size=3)
    x_train += augmented_data[0]
    y_train += augmented_data[1]

    print('Save dataset..')
    dataset.save_dataset(images_path=UPDATED_IMAGES_PATH,
                         labels_path=UPDATED_LABELS_PATH)

    # smile_detector = SVMSmileDetector()
    smile_detector = CNNSmileDetector(input_shape=(*IMAGE_SIZE, IMAGE_CHANNELS))

    print('Train model..')
    smile_detector.train(x_train, y_train, x_val, y_val)

    print('Test model..')
    smile_detector.test(x_test, y_test)

    print('Save model..')
    smile_detector.save_model(MODELS_PATH)

    print('Load model..')
    smile_detector.load_model(models_path=MODELS_PATH)

    print('Start prediction..')
    live_prediction = LivePrediction(smile_detector=smile_detector, image_size=IMAGE_SIZE, image_mode=IMAGE_MODE)
    live_prediction.run(source=VIDEO_PATH)


if __name__ == '__main__':
    main()
