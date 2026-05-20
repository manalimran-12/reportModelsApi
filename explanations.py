"""
Explainable AI helpers:
- Grad-CAM for the pneumonia CNN
- SHAP-based feature attribution for the tabular sklearn models
"""
import os
import numpy as np
import cv2

import matplotlib
matplotlib.use("Agg")  # headless — no display needed
import matplotlib.cm as cm

try:
    import tensorflow as tf
except Exception:
    tf = None

try:
    import shap
except Exception:
    shap = None


# ---------- Grad-CAM (pneumonia) ----------

def _find_last_conv_layer(model):
    """Walk layers in reverse, return the name of the last Conv2D."""
    if tf is None:
        return None
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    return None


def compute_gradcam_heatmap(model, preprocessed_tensor, predicted_class):
    """
    Returns a 2D numpy array (values 0..1) at the feature-map resolution.
    Returns None if the model has no conv layers or computation fails.
    """
    if tf is None:
        return None
    try:
        layer_name = _find_last_conv_layer(model)
        if not layer_name:
            return None

        grad_model = tf.keras.models.Model(
            inputs=model.inputs,
            outputs=[model.get_layer(layer_name).output, model.output],
        )

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(preprocessed_tensor)
            class_output = predictions[:, int(predicted_class)]

        grads = tape.gradient(class_output, conv_outputs)
        if grads is None:
            return None

        # Average the gradients over spatial dims to get per-channel weights
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))  # shape: (C,)

        conv_outputs = conv_outputs[0]                         # (H, W, C)
        heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)  # (H, W)
        heatmap = tf.maximum(heatmap, 0)
        max_val = tf.reduce_max(heatmap)
        if max_val > 0:
            heatmap = heatmap / max_val
        return heatmap.numpy()
    except Exception as e:
        print(f"[gradcam] failed: {e}")
        return None


def save_gradcam_overlay(original_img_path, heatmap, output_path, intensity=0.45):
    """
    Renders a colored Grad-CAM overlay on top of the original image and
    saves it to output_path. Returns True on success.
    """
    try:
        img = cv2.imread(original_img_path)
        if img is None:
            return False
        h, w = img.shape[:2]
        heatmap_resized = cv2.resize(heatmap, (w, h))
        colored = cm.jet(heatmap_resized)[:, :, :3]            # drop alpha
        colored = (colored * 255).astype(np.uint8)
        colored_bgr = cv2.cvtColor(colored, cv2.COLOR_RGB2BGR)
        overlay = cv2.addWeighted(img, 1.0 - intensity, colored_bgr, intensity, 0)
        cv2.imwrite(output_path, overlay)
        return True
    except Exception as e:
        print(f"[gradcam] overlay write failed: {e}")
        return False


# ---------- SHAP (breast / heart / liver) ----------

def compute_shap_explanation(model, input_df, top_k=6, present_features=None):
    """
    Returns a list of {feature, value, shapValue, effect} sorted by |shapValue|.
    Falls back gracefully to None if SHAP can't explain this model.

    Only features that were actually present in the input (non-zero after OCR)
    are reported — missing features default to 0 in the pipeline and would
    otherwise crowd the explanation with meaningless entries.
    """
    if shap is None:
        return None
    try:
        # shap.Explainer auto-selects TreeExplainer/LinearExplainer/etc.
        explainer = shap.Explainer(model)
        shap_values = explainer(input_df)

        vals = np.array(shap_values.values)[0]  # first (only) sample
        # Multi-output: for binary classification shap may return (n_features, 2)
        if vals.ndim > 1:
            vals = vals[:, 1] if vals.shape[-1] == 2 else vals[:, 0]

        features = list(input_df.columns)
        raw_values = input_df.iloc[0].tolist()

        present_feature_set = set(present_features or [])
        rows = []
        for i, f in enumerate(features):
            v = float(raw_values[i])
            sv = float(vals[i])
            if present_feature_set and f not in present_feature_set:
                continue
            # Skip features that were missing (0) AND had near-zero SHAP contribution
            if v == 0 and abs(sv) < 1e-6:
                continue
            rows.append({
                "feature": f,
                "value": v,
                "shapValue": sv,
                "effect": "positive" if sv > 0 else "negative",
            })
        rows.sort(key=lambda r: abs(r["shapValue"]), reverse=True)
        return rows[:top_k]
    except Exception as e:
        print(f"[shap] failed: {e}")
        return None
