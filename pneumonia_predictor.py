import cv2
import numpy as np
import tensorflow as tf
import os
from tensorflow.keras.preprocessing import image

class PneumoniaPredictor:
    def __init__(self, model_path='models/pneumonia.h5', img_size=(36, 36)):
        try:
            self.model = tf.keras.models.load_model(model_path)
            self.img_size = img_size
            self.class_names = ['Normal', 'Pneumonia']
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise

    def preprocess_image(self, img_path):
        try:
            # Load the image
            img = cv2.imread(img_path)
            if img is None:
                raise ValueError(f"Failed to load image from {img_path}")
                
            # Convert to grayscale if needed
            if len(img.shape) == 3:  # If RGB
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            elif len(img.shape) == 4:  # If RGBA
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
            
            # Resize image
            img = cv2.resize(img, self.img_size)
            
            # Convert to float and normalize
            img_array = img.astype('float32') / 255.0
            
            # Add channel dimension
            img_array = np.expand_dims(img_array, axis=-1)
            
            # Add batch dimension
            img_array = np.expand_dims(img_array, axis=0)
            
            return img_array
        except Exception as e:
            print(f"Error preprocessing image: {str(e)}")
            return None

    def predict(self, img_path):
        try:
            # Preprocess the image
            img_array = self.preprocess_image(img_path)
            if img_array is None:
                return None

            # Make prediction
            predictions = self.model.predict(img_array)
            predicted_class = np.argmax(predictions, axis=1)[0]
            confidence = predictions[0][predicted_class] * 100
            
            result = self.class_names[predicted_class]
            return result, confidence
            
        except Exception as e:
            print(f"Error during prediction: {str(e)}")
            return None, 0.0
