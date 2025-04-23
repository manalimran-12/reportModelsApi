# X-ray Prediction API

A FastAPI-based backend service for predicting pneumonia from X-ray images.

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
```

2. Activate the virtual environment:
- Windows:
```bash
venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Copy the pneumonia model file (`pneumonia.h5`) to a `models` directory:
```bash
mkdir models
# Copy your pneumonia.h5 model file to models directory
```

## Running the API

Start the FastAPI server:
```bash
python main.py
```

The server will run on `http://localhost:8000`

## API Endpoints

- `GET /` - Welcome message
- `POST /predict/` - Upload X-ray image for prediction

## Example Usage

Using curl:
```bash
curl -X POST "http://localhost:8000/predict/" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@path/to/your/xray.jpg"
```

## Response Format

```json
{
    "prediction": "Normal" or "Pneumonia",
    "confidence": 0.0 to 100.0,
    "message": "Prediction successful"
}
```
