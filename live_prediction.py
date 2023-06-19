import threading

import cv2

from face_detector import FaceDetector
from image_util import ImageUtil


class LivePrediction:
    def __init__(self, smile_detector):
        self.face_detector = FaceDetector()
        self.smile_detector = smile_detector
        self.face_locations = []
        self.detect_thread = None

    def run(self, source):
        """Run the live prediction on a video source."""
        cap = cv2.VideoCapture(source)
        while True:
            ret, frame = cap.read()

            if not ret:
                cv2.waitKey(0)
                break

            self._predict_and_display_smile(frame)
            cv2.imshow('Smile Detector', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    def _start_face_detection_thread(self, frame):
        """Start a new thread to detect faces in a frame."""
        self.detect_thread = threading.Thread(target=self._detect_faces, args=(frame,))
        self.detect_thread.start()

    def _detect_faces(self, frame):
        """Detect faces in a frame."""
        self.face_locations = self.face_detector.detect_faces(frame, method='fr')

    def _predict_and_display_smile(self, frame):
        """Predict and display smile label for each face in the frame."""
        if len(self.face_locations) == 0:
            self._detect_faces(frame)
        elif self.detect_thread is None or not self.detect_thread.is_alive():
            self._start_face_detection_thread(frame)

        detected_faces = []

        for face_box in self.face_locations:
            top, right, bottom, left = face_box
            face = ImageUtil.convert_to_gray(frame[top:bottom, left:right])
            face = ImageUtil.resize_image(face, (128, 128))
            detected_faces.append((face_box, face))

        for face_box, face in detected_faces:
            prediction = self.smile_detector.predict(face)
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
        red_color = (0, 0, 255)
        yellow_color = (0, 255, 255)
        green_color = (0, 255, 0)

        text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)

        padding = 5

        cv2.rectangle(frame, (left, top), (right, bottom), green_color, 2)

        left = int(left - padding)
        top = int(top - text_size[1] - 2 * padding)
        right = int(left + text_size[0] + 2 * padding)
        bottom = int(top + text_size[1] + 2 * padding)

        cv2.rectangle(frame, (left, top), (right, bottom), yellow_color, cv2.FILLED)

        cv2.putText(frame, text, (left + padding, top + text_size[1] + padding), font, font_scale, red_color,
                    thickness, cv2.LINE_AA, False)
