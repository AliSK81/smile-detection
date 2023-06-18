from cnn_smile_detector import CNNSmileDetector
from genki4_dataset import Genki4Dataset
from live_prediction import LivePrediction

IMAGES_PATH = 'resources/genki4k/files'
LABELS_PATH = 'resources/genki4k/labels.txt'
UPDATED_IMAGES_PATH = 'resources/genki4k_updated/files'
UPDATED_LABELS_PATH = 'resources/genki4k_updated/labels.txt'
MODELS_PATH = 'resources/models'
VIDEO_PATH = 1  # webcam


def main():
    print('Load dataset..')
    dataset = Genki4Dataset(drop_bad_images=True,
                            resize_images=False,
                            crop_faces=False)

    dataset.load_dataset(images_path=UPDATED_IMAGES_PATH,
                         labels_path=UPDATED_LABELS_PATH)

    print('Save dataset..')
    dataset.save_dataset(images_path=UPDATED_IMAGES_PATH,
                         labels_path=UPDATED_LABELS_PATH)

    smile_detector = CNNSmileDetector()

    print('Train model..')
    smile_detector.train_and_test(dataset.images, dataset.labels)

    print('Save model..')
    smile_detector.save_model(MODELS_PATH)

    print('Load model..')
    smile_detector.load_model(models_path=MODELS_PATH)

    live_prediction = LivePrediction(smile_detector=smile_detector)
    live_prediction.run(source=VIDEO_PATH)


if __name__ == '__main__':
    main()
