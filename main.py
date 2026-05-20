from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Any, Dict
import os
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


@app.get("/")
def read_root():
    return {"message": "Welcome to Medical Prediction API"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
