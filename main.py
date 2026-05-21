from fastapi import FastAPI, HTTPException
from fastapi import File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Any, Dict
import base64
import os
import tempfile
from pneumonia_predictor import PneumoniaPredictor
from breast_predictor import BreastCancerPredictor
from heart_predictor import HeartDiseasePredictor
from liver_predictor import LiverDiseasePredictor
from ocr_utils import OCRConfigurationError

app = FastAPI(title="Medical Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictionResponse(BaseModel):
    prediction: str
    confidence: Optional[float] = None
    message: str
    explanation: Optional[Dict[str, Any]] = None


class MedicalPredictionRequest(BaseModel):
    image_path: str
    model_type: str  # 'breast', 'heart', 'liver'


class PneumoniaPredictionRequest(BaseModel):
    image_path: str


def _attach_gradcam_data_url(explanation: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not explanation or explanation.get("type") != "gradcam":
        return explanation

    image_path = explanation.get("imagePath")
    if not image_path or not os.path.exists(image_path):
        return explanation

    with open(image_path, "rb") as f:
        image_data_url = "data:image/png;base64," + base64.b64encode(f.read()).decode("ascii")

    return {
        "type": "gradcam",
        "imageDataUrl": image_data_url,
        "description": explanation.get("description"),
    }


async def _save_upload_to_temp(upload: UploadFile) -> str:
    suffix = os.path.splitext(upload.filename or "")[1] or ".img"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await upload.read())
        return tmp.name


def _cleanup_prediction_files(img_path: str, explanation: Optional[Dict[str, Any]] = None) -> None:
    paths = [img_path]
    if explanation and explanation.get("type") == "gradcam":
        image_path = explanation.get("imagePath")
        if image_path:
            paths.append(image_path)

    for path in paths:
        try:
            if path and os.path.exists(path):
                os.remove(path)
        except OSError:
            pass


@app.post("/predict/pneumonia/", response_model=PredictionResponse)
async def predict_pneumonia(request: PneumoniaPredictionRequest):
    try:
        img_path = os.path.abspath(os.path.expanduser(request.image_path))
        if not os.path.exists(img_path):
            raise HTTPException(
                status_code=400,
                detail=f"File not found at the specified path: {img_path}",
            )

        predictor = PneumoniaPredictor()
        result = predictor.predict(img_path)
        # Predictor returns either None or a 3-tuple. Treat (None, None, None) as failure too.
        if result is None or result[0] is None:
            raise HTTPException(
                status_code=422,
                detail="Model could not process this input. See server logs for details.",
            )

        prediction, confidence, explanation = result
        return PredictionResponse(
            prediction=prediction,
            confidence=confidence,
            message="Pneumonia prediction successful",
            explanation=explanation,
        )
    except HTTPException:
        raise
    except OCRConfigurationError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/pneumonia/upload/", response_model=PredictionResponse)
async def predict_pneumonia_upload(file: UploadFile = File(...)):
    img_path = await _save_upload_to_temp(file)
    explanation = None
    try:
        predictor = PneumoniaPredictor()
        result = predictor.predict(img_path)
        if result is None or result[0] is None:
            raise HTTPException(
                status_code=422,
                detail="Model could not process this input. See server logs for details.",
            )

        prediction, confidence, explanation = result
        return PredictionResponse(
            prediction=prediction,
            confidence=confidence,
            message="Pneumonia prediction successful",
            explanation=_attach_gradcam_data_url(explanation),
        )
    except HTTPException:
        raise
    except OCRConfigurationError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        _cleanup_prediction_files(img_path, explanation)


@app.post("/predict/medical/", response_model=PredictionResponse)
async def predict_medical(request: MedicalPredictionRequest):
    try:
        img_path = os.path.abspath(os.path.expanduser(request.image_path))
        if not os.path.exists(img_path):
            raise HTTPException(
                status_code=400,
                detail=f"File not found at the specified path: {img_path}",
            )

        if request.model_type == 'breast':
            predictor = BreastCancerPredictor()
        elif request.model_type == 'heart':
            predictor = HeartDiseasePredictor()
        elif request.model_type == 'liver':
            predictor = LiverDiseasePredictor()
        else:
            raise HTTPException(
                status_code=400,
                detail="Invalid model_type. Must be one of: breast, heart, liver",
            )

        result = predictor.predict(img_path)
        # Predictor returns either None or a 3-tuple. Treat (None, None, None) as failure too.
        if result is None or result[0] is None:
            raise HTTPException(
                status_code=422,
                detail="Model could not process this input. See server logs for details.",
            )

        prediction, confidence, explanation = result
        return PredictionResponse(
            prediction=prediction,
            confidence=confidence,
            message=f"{request.model_type.capitalize()} prediction successful",
            explanation=explanation,
        )
    except HTTPException:
        raise
    except OCRConfigurationError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/medical/upload/", response_model=PredictionResponse)
async def predict_medical_upload(
    model_type: str = Form(...),
    file: UploadFile = File(...),
):
    img_path = await _save_upload_to_temp(file)
    try:
        if model_type == 'breast':
            predictor = BreastCancerPredictor()
        elif model_type == 'heart':
            predictor = HeartDiseasePredictor()
        elif model_type == 'liver':
            predictor = LiverDiseasePredictor()
        else:
            raise HTTPException(
                status_code=400,
                detail="Invalid model_type. Must be one of: breast, heart, liver",
            )

        result = predictor.predict(img_path)
        if result is None or result[0] is None:
            raise HTTPException(
                status_code=422,
                detail="Model could not process this input. See server logs for details.",
            )

        prediction, confidence, explanation = result
        return PredictionResponse(
            prediction=prediction,
            confidence=confidence,
            message=f"{model_type.capitalize()} prediction successful",
            explanation=explanation,
        )
    except HTTPException:
        raise
    except OCRConfigurationError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        _cleanup_prediction_files(img_path)


@app.get("/")
def read_root():
    return {"message": "Welcome to Medical Prediction API"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
