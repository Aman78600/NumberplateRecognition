import cv2
import pickle
import warnings
from ultralytics import YOLO
import numpy as np
from paddleocr import PaddleOCR
import streamlit as st
import tempfile
import logging
import os

os.PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION='python'
# Suppress specific deprecation warnings
from cryptography.utils import CryptographyDeprecationWarning

logging.getLogger('ppocr').setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=CryptographyDeprecationWarning)

# Initialize models and classes
class CarDetection:
    def __init__(self):
        self.model = YOLO("yolov10x.pt")

    def detect_and_process_cars(self, frame):
        result = self.model(frame, classes=[2])  # Class 2 corresponds to cars
        car_crops = []

        for box in result[0].boxes:
            conf = float(box.conf)
            if conf >= 0.55:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                car_crop = frame[y1:y2, x1:x2]
                car_crops.append(car_crop)

        return car_crops


class LicensePlateDetector:
    def __init__(self):
        self.model = None

    def initialize_model(self):
        with open("license_plate_model.pkl", "rb") as file:
            self.model = pickle.load(file)

    def detect_license_plate(self, image, confidence=0, overlap=40):
        if self.model is None:
            raise Exception("Model not initialized! Please initialize the model first.")
        predictions = self.model.predict(image, confidence=confidence, overlap=overlap).json()
        for prediction in predictions['predictions']:
            x, y, width, height = prediction['x'], prediction['y'], prediction['width'], prediction['height']
            x1, y1 = int(x - width / 2), int(y - height / 2)
            x2, y2 = int(x + width / 2), int(y + height / 2)
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(image.shape[1], x2), min(image.shape[0], y2)

            if y2 > y1 and x2 > x1:
                return image[y1:y2, x1:x2]
        return None


class ImageTextDetector:
    def __init__(self):
        self.ocr = PaddleOCR(lang='en')

    def detect_text(self, img):
        result = self.ocr.ocr(img, det=False, cls=False)
        return result


# Initialize instances
car_detector = CarDetection()
license_plate_detector = LicensePlateDetector()
license_plate_detector.initialize_model()
text_detector = ImageTextDetector()


# Streamlit app
st.title("Number Plate Detection")
st.write("Upload an image or video to detect number plates.")

# Input type selection
input_type = st.radio("Choose Input Type:", ("Image", "Video"))

if input_type == "Image":
    uploaded_image = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
    if uploaded_image is not None:
        # Process the uploaded image
        file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        cars = car_detector.detect_and_process_cars(image)
        result_image = image.copy()

        st.write(f"Detected {len(cars)} car(s) in the image.")
        for car in cars:
            license_plate = license_plate_detector.detect_license_plate(car)
            if license_plate is not None:
                text_result = text_detector.detect_text(license_plate)
                if text_result and text_result[0][0][0]:
                    x1, y1, x2, y2 = cv2.boundingRect(np.array([[0, 0], [car.shape[1], car.shape[0]]]))
                    cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(result_image, text_result[0][0][0], (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Display the processed image
        st.image(result_image, channels="BGR", caption="Processed Image")

elif input_type == "Video":
    uploaded_video = st.file_uploader("Upload a Video", type=["mp4", "avi", "mov"])
    if uploaded_video is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())
        cap = cv2.VideoCapture(tfile.name)

        stframe = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            cars = car_detector.detect_and_process_cars(frame)
            for car in cars:
                license_plate = license_plate_detector.detect_license_plate(car)
                if license_plate is not None:
                    text_result = text_detector.detect_text(license_plate)
                    if text_result and text_result[0][0][0]:
                        x1, y1, x2, y2 = cv2.boundingRect(np.array([[0, 0], [car.shape[1], car.shape[0]]]))
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, text_result[0][0][0], (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # Display the video frame with detections
            stframe.image(frame, channels="BGR")

        cap.release()
