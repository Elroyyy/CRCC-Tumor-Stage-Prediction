from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Optional, Any, Union
import os
from ccrc import CCRCModel

app = FastAPI(title="CCRC Cancer Stage Classifier API")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize model
ccrc_model = CCRCModel()
ccrc_model.initialize()


# Models for request/response
class PredictionRequest(BaseModel):
    name: str
    features: Dict[str, Any]


class PredictionResponse(BaseModel):
    predicted_stage: str
    description: str
    patient_name: str


# Stage descriptions
STAGE_DESCRIPTIONS = {
    "Stage I": (
        "Stage I CCRC means the tumor is small (usually less than 7 cm) "
        "and limited to the kidney. At this stage, the cancer has not "
        "spread beyond the kidney. Treatment is often very effective, "
        "and surgery may fully remove the tumor."
    ),
    "Stage II": (
        "Stage II CCRC indicates that the tumor is larger than in Stage I "
        "but is still confined within the kidney. While it is bigger in size, "
        "it has not spread to nearby lymph nodes or distant organs. "
        "Treatment usually focuses on removing the tumor surgically."
    ),
    "Stage III": (
        "Stage III CCRC means the cancer may have spread into nearby major "
        "blood vessels or lymph nodes, but it is still relatively local. "
        "This stage shows greater progression than Stage II and often requires "
        "a combination of treatments, such as surgery and additional therapies."
    ),
    "Stage IV": (
        "Stage IV CCRC is the most advanced stage, where the cancer has spread "
        "beyond the kidney to distant lymph nodes or other organs (such as lungs, "
        "bones, or liver). Treatment at this stage may include surgery, targeted "
        "therapy, immunotherapy, or a combination, depending on the patient's health."
    ),
}

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", response_class=HTMLResponse)
async def serve_home():
    """Serve the main HTML page"""
    return FileResponse("static/index.html")


@app.get("/api/features")
async def get_features():
    """Get all features and their options"""
    features_info = []

    for feature in ccrc_model.features:
        feature_info = {
            "name": feature,
            "type": "categorical" if feature in ccrc_model.label_encoders else "numerical"
        }

        if feature in ccrc_model.label_encoders:
            feature_info["options"] = ccrc_model.get_feature_options(feature)

        features_info.append(feature_info)

    return {"features": features_info}


@app.post("/api/predict", response_model=PredictionResponse)
async def predict_stage(request: PredictionRequest):
    """Predict cancer stage"""
    try:
        if not request.name.strip():
            raise HTTPException(status_code=400, detail="Patient name is required")

        # Encode input features
        encoded_features = {}
        for feature in ccrc_model.features:
            if feature not in request.features:
                raise HTTPException(
                    status_code=400,
                    detail=f"Missing feature: {feature}"
                )

            value = request.features[feature]
            encoded_features[feature] = ccrc_model.encode_input(feature, value)

        # Make prediction
        predicted_stage = ccrc_model.predict(encoded_features)

        # Save to database
        ccrc_model.save_prediction(request.name, predicted_stage, encoded_features)

        # Get description
        description = STAGE_DESCRIPTIONS.get(
            predicted_stage,
            "No description available."
        )

        return PredictionResponse(
            predicted_stage=predicted_stage,
            description=description,
            patient_name=request.name
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/records")
async def get_records():
    """Get all saved patient records"""
    try:
        records = ccrc_model.get_saved_records()
        return {"records": records}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": ccrc_model.model is not None,
        "features_count": len(ccrc_model.features)
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)