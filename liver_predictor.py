import cv2
import pytesseract
import pandas as pd
import numpy as np
import pickle
import re
from sklearn.preprocessing import StandardScaler
from explanations import compute_shap_explanation
from ocr_utils import OCRConfigurationError, configure_tesseract

class LiverDiseasePredictor:
    def __init__(self, model_path='models/liver.pkl'):
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)

        self.required_features = [
            'Age', 'Gender', 'Total_Bilirubin', 'Direct_Bilirubin', 'Alkaline_Phosphotase',
            'Alamine_Aminotransferase', 'Aspartate_Aminotransferase', 'Total_Protiens',
            'Albumin', 'Albumin_and_Globulin_Ratio'
        ]
        self.min_feature_count = 7

    def _normalize_ocr_text(self, text):
        text = text.lower()
        text = text.replace("|", " ")
        text = re.sub(r'[^\w\s\.:/\-]', ' ', text)
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n+', '\n', text)
        return text.strip()

    def _extract_numeric_after_label(self, line, label_pattern):
        match = re.search(label_pattern, line)
        if not match:
            return None

        tail = line[match.end():]
        numbers = re.findall(r'\d+(?:\.\d+)?', tail)
        if not numbers:
            return None

        # Lab rows usually contain result first and reference range after it.
        # We prefer the first number unless it looks like a range fragment.
        return float(numbers[0])

    def _extract_age_gender(self, text):
        features = {}
        match = re.search(r'age\s*/?\s*gender\s*:?\s*(\d+(?:\.\d+)?)\s*/?\s*(male|female)', text)
        if match:
            features['Age'] = float(match.group(1))
            features['Gender'] = 1 if match.group(2) == 'male' else 0
        return features

    def _extract_liver_panel_features(self, text):
        features = self._extract_age_gender(text)

        line_patterns = {
            'Total_Bilirubin': r'(?:bilirubin|cilirubin)\s+total',
            'Direct_Bilirubin': r'(?:bilirubin|cilirubin)\s+direct',
            'Alamine_Aminotransferase': r'sgpt|alt',
            'Aspartate_Aminotransferase': r'sgot|ast',
            'Alkaline_Phosphotase': r'alkaline\s+phosphatase',
            'Total_Protiens': r'total\s+proteins?',
            'Albumin': r'albumin|aldumsn|atbumin',
            'Albumin_and_Globulin_Ratio': r'a\s*:?\s*g\s*ratio|a:g\s*ratio',
        }

        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                continue

            for feature_name, label_pattern in line_patterns.items():
                if feature_name in features:
                    continue
                value = self._extract_numeric_after_label(line, label_pattern)
                if value is None:
                    continue

                if feature_name == 'Total_Protiens' and value > 10:
                    value = value / 10

                features[feature_name] = value

        return features

    def preprocess_image(self, image_path):
        try:
            configure_tesseract()
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError("Could not read image file")

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            upscaled = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            thresh = cv2.threshold(upscaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            config = r'--oem 3 --psm 6 -l eng'
            text = pytesseract.image_to_string(thresh, config=config)
            return self._normalize_ocr_text(text)
        except OCRConfigurationError:
            raise
        except Exception as e:
            print(f"Error processing image: {str(e)}")
            return None

    def _extract_feature_value(self, text, feature_name, used_values):
        try:
            features = self._extract_liver_panel_features(text)
            val = features.get(feature_name)
            if val is not None and val not in used_values:
                used_values.add(val)
                return val
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

            features = self._extract_liver_panel_features(text)

            if not features:
                raise ValueError("No features could be extracted from the image")
            if len(features) < self.min_feature_count:
                raise ValueError(
                    f"Only extracted {len(features)} of {len(self.required_features)} required liver features"
                )

            input_df = self.convert_to_model_input(features)
            if input_df is None:
                raise ValueError("Could not convert features to model input format")

            prediction = self.model.predict(input_df)[0]
            if hasattr(self.model, 'predict_proba'):
                proba = self.model.predict_proba(input_df)[0]
                confidence = float(proba[1]) if len(proba) > 1 else float(proba[0])
            else:
                confidence = None

            result = "Liver Disease" if prediction == 1 else "No Liver Disease"

            shap_rows = compute_shap_explanation(
                self.model,
                input_df,
                top_k=6,
                present_features=features.keys(),
            )
            explanation = None
            if shap_rows:
                explanation = {
                    "type": "shap",
                    "features": shap_rows,
                    "description": (
                        "Each bar shows how much a liver panel value pushed the prediction toward "
                        "Liver Disease (positive) or No Liver Disease (negative)."
                    ),
                }
            return result, confidence, explanation
        except OCRConfigurationError:
            raise
        except Exception as e:
            print(f"Error during prediction: {str(e)}")
            return None, None, None
