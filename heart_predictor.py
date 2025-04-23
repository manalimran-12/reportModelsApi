import cv2
import pytesseract
import pandas as pd
import numpy as np
import pickle
import re
import os

# Configure Tesseract path
TESSERACT_PATH = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

class HeartDiseasePredictor:
    def __init__(self, model_path='models/heart.pkl'):
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)

        self.required_features = [
            'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
            'restecg', 'thalach', 'exang', 'oldpeak', 'slope',
            'ca', 'thal'
        ]

    def preprocess_image(self, image_path):
        try:
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError("Could not read image file")

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           cv2.THRESH_BINARY, 11, 2)
            kernel = np.ones((2, 2), np.uint8)
            dilated = cv2.dilate(thresh, kernel, iterations=1)
            config = r'--oem 3 --psm 6 -l eng'
            text = pytesseract.image_to_string(dilated, config=config)

            text = text.lower()
            text = re.sub(r'[^a-zA-Z0-9\s\.:]', ' ', text)
            text = re.sub(r'\s+', ' ', text).strip()

            return text
        except Exception as e:
            print(f"Error processing image: {str(e)}")
            return None

    def _extract_feature_value(self, text, feature_name, used_values):
        try:
            # Define regex patterns for each feature
            patterns = {
                'age': [r'age[\s:]*([\d.]+)'],
                'sex': [r'sex[\s:]*([\d.]+)', r'male', r'female'],
                'cp': [r'cp[\s:]*([\d.]+)', r'chest pain[\s:]*([\d.]+)'],
                'trestbps': [r'trestbps[\s:]*([\d.]+)', r'blood pressure[\s:]*([\d.]+)'],
                'chol': [r'chol[\s:]*([\d.]+)', r'cholesterol[\s:]*([\d.]+)'],
                'fbs': [r'fbs[\s:]*([\d.]+)', r'fasting blood[\s:]*([\d.]+)'],
                'restecg': [r'restecg[\s:]*([\d.]+)'],
                'thalach': [r'thalach[\s:]*([\d.]+)', r'max heart rate[\s:]*([\d.]+)'],
                'exang': [r'exang[\s:]*([\d.]+)', r'exercise angina[\s:]*([\d.]+)'],
                'oldpeak': [r'oldpeak[\s:]*([\d.]+)'],
                'slope': [r'slope[\s:]*([\d.]+)'],
                'ca': [r'ca[\s:]*([\d.]+)'],
                'thal': [r'thal[\s:]*([\d.]+)']
            }

            if feature_name == 'sex':
                if 'male' in text:
                    return 1
                elif 'female' in text:
                    return 0

            for pattern in patterns.get(feature_name, []):
                matches = re.findall(pattern, text)
                for match in matches:
                    try:
                        val = float(match) if isinstance(match, str) else float(match[0])
                        if val not in used_values:
                            used_values.add(val)
                            return val
                    except:
                        continue

            return None
        except Exception as e:
            print(f"Error extracting feature {feature_name}: {str(e)}")
            return None

    def convert_to_model_input(self, features):
        try:
            df = pd.DataFrame([{f: features.get(f, 0.0) for f in self.required_features}])
            df = df.fillna(0.0)
            return df
        except Exception as e:
            print(f"Error converting to model input: {str(e)}")
            return None

    def predict(self, image_path):
        try:
            text = self.preprocess_image(image_path)
            if not text:
                raise ValueError("Could not extract text from image")

            features = {}
            used_values = set()
            for feature in self.required_features:
                value = self._extract_feature_value(text, feature, used_values)
                if value is not None:
                    features[feature] = value

            if not features:
                raise ValueError("No features could be extracted from the image")

            input_df = self.convert_to_model_input(features)
            if input_df is None:
                raise ValueError("Could not convert features to model input format")

            prediction = self.model.predict(input_df)
            result = "Heart Disease" if prediction[0] == 1 else "No Heart Disease"
            return result
        except Exception as e:
            print(f"Error during prediction: {str(e)}")
            return None
