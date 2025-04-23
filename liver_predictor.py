import cv2
import pytesseract
import pandas as pd
import numpy as np
import pickle
import re
import os
from sklearn.preprocessing import StandardScaler
import joblib
import tempfile
import shutil

# Configure Tesseract path
TESSERACT_PATH = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

class LiverDiseasePredictor:
   def __init__(self, model_path='models/liver.pkl'):
    try:
        # First try loading with joblib which is safer for sklearn models
        try:
            self.model = joblib.load(model_path)
        except Exception as joblib_error:
            print(f"Joblib loading failed: {str(joblib_error)}")
            
            # If joblib fails, try pickle with sklearn version compatibility fix
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
                
                # Handle scikit-learn version incompatibility
                if hasattr(model_data, 'tree_'):
                    try:
                        # Check if tree has the expected structure
                        tree_node_dtype = model_data.tree_.__getstate__()['nodes'].dtype
                        
                        # If missing the 'missing_go_to_left' field, we need to add it
                        if 'missing_go_to_left' not in tree_node_dtype.names:
                            print("Fixing missing field in decision tree nodes")
                            
                            # Get the original nodes
                            nodes = model_data.tree_.__getstate__()['nodes']
                            
                            # Create array with new dtype that includes missing field
                            new_dtype = np.dtype([
                                ('left_child', '<i8'),
                                ('right_child', '<i8'),
                                ('feature', '<i8'),
                                ('threshold', '<f8'),
                                ('impurity', '<f8'),
                                ('n_node_samples', '<i8'),
                                ('weighted_n_node_samples', '<f8'),
                                ('missing_go_to_left', 'u1')
                            ])
                            
                            # Create new nodes array with updated dtype
                            new_nodes = np.zeros(nodes.shape, dtype=new_dtype)
                            
                            # Copy values from old array to new array
                            for field in nodes.dtype.names:
                                new_nodes[field] = nodes[field]
                                
                            # Set default value for missing field (usually False/0)
                            new_nodes['missing_go_to_left'] = 0
                            
                            # Update tree nodes
                            model_data.tree_.__getstate__()['nodes'] = new_nodes
                    except Exception as dtype_err:
                        print(f"Error fixing dtype: {str(dtype_err)}")
                
                self.model = model_data

        self.required_features = [
            'Age', 'Gender', 'Total_Bilirubin', 'Direct_Bilirubin', 'Alkaline_Phosphotase',
            'Alamine_Aminotransferase', 'Aspartate_Aminotransferase', 'Total_Protiens',
            'Albumin', 'Albumin_and_Globulin_Ratio'
        ]
        
        # Create a mapping for gender
        self.gender_mapping = {'Male': 1, 'Female': 0}
        
    except Exception as e:
        print(f"Error initializing predictor: {str(e)}")
        raise
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
            text = text.lower()
            text = re.sub(r'[^a-zA-Z0-9\s\.:]', ' ', text)
            text = re.sub(r'\s+', ' ', text).strip()

            # Special case for Gender
            if feature_name == 'Gender':
                if 'male' in text:
                    return 1
                elif 'female' in text:
                    return 0
                else:
                    return None

            # Improved fuzzy regex map
            patterns = {
                'Age': [r'age[\s:]*([\d.]+)'],
                'Total_Bilirubin': [r'total.*bill?rubin[\s:]*([\d.]+)'],
                'Direct_Bilirubin': [
                    r'direct.*bill?rubin[\s:]*([\d.]+)',
                    r'total.*bill?rubin.*?([\d.]+)\s+([\d.]+)'  # fallback to get 2nd number
                ],
                'Alkaline_Phosphotase': [r'alkaline.*phosph[ao]tase[\s:]*([\d.]+)'],
                'Alamine_Aminotransferase': [r'alamine.*aminotransferase[\s:]*([\d.]+)'],
                'Aspartate_Aminotransferase': [
                    r'aspartate.*aminotransferase[\s:]*([\d.]+)',
                    r'alt.*sgpt[\s:]*([\d.]+)'
                ],
                'Total_Protiens': [r'total.*proteins[\s:]*([\d.]+)', r'total.*protiens[\s:]*([\d.]+)'],
                'Albumin': [r'albumin[\s:]*([\d.]+)', r'atbumin[\s:]*([\d.]+)'],
                'Albumin_and_Globulin_Ratio': [
                    r'albumin.*globulin.*ratio[\s:]*([\d.]+)',
                    r'globulin.*ratio[\s:]*([\d.]+)'
                ]
            }

            for pattern in patterns.get(feature_name, []):
                matches = re.findall(pattern, text)
                for match in matches:
                    try:
                        # If pattern returns a tuple (like for two numbers), take second one
                        if isinstance(match, tuple):
                            val = float(match[1])
                        else:
                            val = float(match)

                        # Fix over-detected protein like "74" -> "7.4"
                        if feature_name == 'Total_Protiens' and val > 10:
                            val = val / 10

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

            # Get raw prediction
            prediction_value = self.model.predict(input_df)[0]
            
            # Get probabilities for confidence score if possible
            confidence = 0.0
            try:
                if hasattr(self.model, 'predict_proba'):
                    probs = self.model.predict_proba(input_df)[0]
                    confidence = probs[1] if prediction_value == 1 else probs[0]
                else:
                    confidence = 0.8  # Fallback confidence if no probabilities
            except:
                confidence = 0.7  # Default confidence on error
        
        # Format the output to match what the API expects
            result = "Liver Disease" if prediction_value == 1 else "No Liver Disease"
            return result, confidence
        
        except Exception as e:
            print(f"Error during prediction: {str(e)}")
            return None