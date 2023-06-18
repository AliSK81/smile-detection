from feature_extractor import FeatureExtractor
from genki4_dataset import Genki4Dataset
from live_prediction import LivePrediction
from smile_detector import SmileDetector

IMAGES_PATH = 'resources/genki4k/files'
LABELS_PATH = 'resources/genki4k/labels.txt'
UPDATED_IMAGES_PATH = 'resources/genki4k_updated/files'
UPDATED_LABELS_PATH = 'resources/genki4k_updated/labels.txt'
MODEL_PATH = 'resources/models/smile_detector.pkl'
VIDEO_PATH = 0  # webcam


def main():
    print('Load dataset..')
    dataset = Genki4Dataset(drop_bad_images=True,
                            resize_images=True,
                            crop_faces=True)

    dataset.load_dataset(images_path=UPDATED_IMAGES_PATH,
                         labels_path=UPDATED_LABELS_PATH)

    print('Save dataset..')
    dataset.save_dataset(images_path=UPDATED_IMAGES_PATH,
                         labels_path=UPDATED_LABELS_PATH)

    print('Extract features..')
    feature_extractor = FeatureExtractor(images=dataset.images)
    features = feature_extractor.extract_features()

    print('Train model..')
    smile_detector = SmileDetector(features, dataset.labels)
    smile_detector.train_and_test()

    print('Save model..')
    smile_detector.save_model(MODEL_PATH)

    live_prediction = LivePrediction(model_path=MODEL_PATH)
    live_prediction.run(source=VIDEO_PATH)


if __name__ == '__main__':
    main()
