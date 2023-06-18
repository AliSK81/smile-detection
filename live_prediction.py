import pickle

import cv2

from face_detector import FaceDetector
from feature_extractor import FeatureExtractor
from image_util import ImageUtil
from logger import Logger


class LivePrediction:
    def __init__(self, model_path):
        self.face_detector = FaceDetector()

        self._load_model(model_path)

    def run(self, source):
        """Run the live prediction on a video source."""
        cap = cv2.VideoCapture(source)
        while True:
            ret, frame = self._get_frame(cap)

            if not ret:
                cv2.waitKey(0)
                break

            self._predict_and_display_smile(frame)
            cv2.imshow('Smile Detector', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

    @Logger.log_time
    def _load_model(self, model_path):
        """Load the model from file."""
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)

    def _get_frame(self, cap):
        """Read a frame from the video source."""
        ret, frame = cap.read()
        return ret, frame

    @Logger.log_time
    def _detect_faces(self, frame):
        """Detect faces in a frame."""
        face_locations = self.face_detector.detect_faces(frame, method='cv2')
        detected_faces = []

        for face_box in face_locations:
            top, right, bottom, left = face_box
            face = ImageUtil.convert_to_gray(frame[top:bottom, left:right])
            face = ImageUtil.resize_image(face, (128, 128))
            feature_extractor = FeatureExtractor([face])
            features = feature_extractor.extract_features()
            detected_faces.append((face_box, features))

        return detected_faces

    @Logger.log_time
    def _predict_and_display_smile(self, frame):
        """Predict and display smile label for each face in the frame."""
        for face_box, features in self._detect_faces(frame):
            prediction = self.model.predict(features)
            self.display_smile_prediction(frame, prediction, face_box)

    def display_smile_prediction(self, frame, prediction, face_box):
        """Display the smile label and emoji on the frame."""
        top, right, bottom, left = face_box
        label = 'Smile' if prediction[0] == 1 else 'No Smile'
        emoji_text = ':)' if prediction[0] == 1 else ''
        text = f'{label} {emoji_text}'
        font = cv2.FONT_HERSHEY_TRIPLEX
        font_scale = 0.5
        thickness = 1
        text_color = (0, 0, 255)
        background_color = (0, 255, 255)

        text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)

        padding = 5
        left = int(left - padding)
        top = int(top - text_size[1] - 2 * padding)
        right = int(left + text_size[0] + 2 * padding)
        bottom = int(top + text_size[1] + 2 * padding)

        cv2.rectangle(frame, (left, top), (right, bottom), background_color, cv2.FILLED)
        cv2.putText(frame, text, (left + padding, top + text_size[1] + padding), font, font_scale, text_color,
                    thickness, cv2.LINE_AA, False)
