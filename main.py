from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import os
from pneumonia_predictor import PneumoniaPredictor
from breast_predictor import BreastCancerPredictor
from heart_predictor import HeartDiseasePredictor
from liver_predictor import LiverDiseasePredictor

app = FastAPI(title="Medical Prediction API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    message: str

class MedicalPredictionRequest(BaseModel):
    image_path: str
    model_type: str  # 'breast', 'heart', 'liver'

class PneumoniaPredictionRequest(BaseModel):
    image_path: str

# Pneumonia Prediction Endpoint
@app.post("/predict/pneumonia/", response_model=PredictionResponse)
async def predict_pneumonia(request: PneumoniaPredictionRequest):
    try:
        # Normalize the path to handle both Windows and Unix-style paths
        img_path = os.path.abspath(os.path.expanduser(request.image_path))
        
        if not os.path.exists(img_path):
            raise HTTPException(
                status_code=400,
                detail=f"File not found at the specified path: {img_path}"
            )
            
        # Initialize predictor
        predictor = PneumoniaPredictor()
        
        # Make prediction
        result = predictor.predict(img_path)
        
        if result is None:
            raise HTTPException(
                status_code=400,
                detail="Failed to process the image"
            )
            
        prediction, confidence = result
        return PredictionResponse(
            prediction=prediction,
            confidence=confidence,
            message="Pneumonia prediction successful"
        )
            
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

# Other Medical Predictions Endpoint
@app.post("/predict/medical/", response_model=PredictionResponse)
async def predict_medical(request: MedicalPredictionRequest):
    try:
        # Normalize the path to handle both Windows and Unix-style paths
        img_path = os.path.abspath(os.path.expanduser(request.image_path))
        
        if not os.path.exists(img_path):
            raise HTTPException(
                status_code=400,
                detail=f"File not found at the specified path: {img_path}"
            )

        # Initialize the appropriate predictor
        if request.model_type == 'breast':
            predictor = BreastCancerPredictor()
        elif request.model_type == 'heart':
            predictor = HeartDiseasePredictor()
        elif request.model_type == 'liver':
            predictor = LiverDiseasePredictor()
        else:
            raise HTTPException(
                status_code=400,
                detail="Invalid model_type. Must be one of: breast, heart, liver"
            )
            
        # Make prediction
        result = predictor.predict(img_path)
        
        if result is None:
            raise HTTPException(
                status_code=400,
                detail="Failed to process the image"
            )
            
        prediction, confidence = result
        return PredictionResponse(
            prediction=prediction,
            confidence=confidence,
            message=f"{request.model_type.capitalize()} prediction successful"
        )
            
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

@app.get("/")
def read_root():
    return {"message": "Welcome to Medical Prediction API"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)