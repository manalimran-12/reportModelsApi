import cv2
import numpy as np
import tensorflow as tf
import os
from explanations import compute_gradcam_heatmap, save_gradcam_overlay

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
            img = cv2.imread(img_path)
            if img is None:
                raise ValueError(f"Failed to load image from {img_path}")

            if len(img.shape) == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            elif len(img.shape) == 4:
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)

            img = cv2.resize(img, self.img_size)
            img_array = img.astype('float32') / 255.0
            img_array = np.expand_dims(img_array, axis=-1)
            img_array = np.expand_dims(img_array, axis=0)
            return img_array
        except Exception as e:
            print(f"Error preprocessing image: {str(e)}")
            return None

    def predict(self, img_path):
        try:
            img_array = self.preprocess_image(img_path)
            if img_array is None:
                return None

            predictions = self.model.predict(img_array)
            predicted_class = int(np.argmax(predictions, axis=1)[0])
            confidence = float(predictions[0][predicted_class] * 100)
            result = self.class_names[predicted_class]

            # Generate Grad-CAM overlay next to the original image
            explanation = None
            heatmap = compute_gradcam_heatmap(self.model, img_array, predicted_class)
            if heatmap is not None:
                base, _ = os.path.splitext(img_path)
                overlay_path = f"{base}_gradcam.png"
                if save_gradcam_overlay(img_path, heatmap, overlay_path):
                    explanation = {
                        "type": "gradcam",
                        "imagePath": overlay_path,
                        "description": (
                            "The highlighted regions show which parts of the X-ray "
                            "most influenced the model's prediction."
                        ),
                    }

            return result, confidence, explanation

        except Exception as e:
            print(f"Error during prediction: {str(e)}")
            return None, 0.0, None