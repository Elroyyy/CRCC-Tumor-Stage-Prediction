from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import Dict, Optional, Any, Union
import os
import jwt
from datetime import datetime, timedelta
from passlib.context import CryptContext
from ccrc import CCRCModel
import requests
from pymongo import MongoClient
import numpy as np
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="CCRC Cancer Stage Classifier API")

# Security
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 480

# Groq API Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama-3.1-8b-instant"

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer()

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
class LoginRequest(BaseModel):
    username: str
    password: str

class LoginResponse(BaseModel):
    access_token: str
    token_type: str
    username: str

class PredictionRequest(BaseModel):
    name: str
    features: Dict[str, Any]


class PredictionResponse(BaseModel):
    predicted_stage: str
    description: str
    patient_name: str


def generate_stage_description_with_groq(predicted_stage: str, patient_data: dict, original_features: dict) -> str:

    # Check if Groq API key is set
    if not GROQ_API_KEY:
        # Fallback to static descriptions if API key not set
        return STAGE_DESCRIPTIONS.get(predicted_stage, "Stage description unavailable.")

    try:
        # Format patient data for LLM context
        patient_info = []
        for feature, value in original_features.items():
            patient_info.append(f"- {feature}: {value}")

        patient_context = "\n".join(patient_info)

        # Create prompt for Groq LLM
        prompt = f"""You are a medical AI assistant explaining Clear Cell Renal Cell Carcinoma (CCRC) stages to patients.

        Patient Data:
        {patient_context}
        
        Predicted Stage: {predicted_stage}
        
        Task: Write a clear, concise explanation (MAXIMUM 4 LINES) of why this patient is classified as {predicted_stage}. 
        
        Requirements:
        1. Reference specific patient data points (e.g., "because the tumor size is X cm")
        2. Explain what this stage means clinically
        3. Be direct and factual
        4. Keep it under 4 lines total
        5. Use clear, professional medical language
        
        Write ONLY the description, nothing else:"""

        # Call Groq API
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": GROQ_MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a medical AI that provides concise, accurate cancer stage explanations. Always keep responses under 4 lines."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.3,  # Lower temperature for more factual responses
            "max_tokens": 250,  # Limit to keep response short
            "top_p": 0.9
        }

        response = requests.post(
            GROQ_API_URL,
            headers=headers,
            json=payload,
            timeout=10
        )

        if response.status_code == 200:
            result = response.json()
            description = result['choices'][0]['message']['content'].strip()

            # Ensure description is not too long (max 4 lines, ~400 chars)
            if len(description) > 400:
                # Truncate and add ellipsis
                description = description[:397] + "..."

            return description
        else:
            print(f"Groq API Error: {response.status_code} - {response.text}")
            # Fallback to static description
            return STAGE_DESCRIPTIONS.get(predicted_stage, "Stage description unavailable.")

    except requests.exceptions.Timeout:
        print("Groq API timeout - using fallback description")
        return STAGE_DESCRIPTIONS.get(predicted_stage, "Stage description unavailable.")
    except Exception as e:
        print(f"Error generating description with Groq: {e}")
        # Fallback to static description
        return STAGE_DESCRIPTIONS.get(predicted_stage, "Stage description unavailable.")

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

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a plain password against a hashed password"""
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """Hash a password using bcrypt"""
    return pwd_context.hash(password)


def create_access_token(data: dict) -> str:
    """Create JWT access token"""
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, algorithm=ALGORITHM)
    return encoded_jwt

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    """Verify JWT token and return username"""
    try:
        token = credentials.credentials
        payload = jwt.decode(token, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials"
            )
        return username
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired. Please login again."
        )
    except jwt.JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials"
        )


def authenticate_user(username: str, password: str):
    """Authenticate user against MongoDB"""
    client = MongoClient('mongodb://localhost:27017/CCRC')
    db = client['patients']
    users_collection = db['users']

    user = users_collection.find_one({"username": username})
    client.close()

    if not user:
        return False
    if not verify_password(password, user['password']):
        return False
    return user


# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")
@app.get("/", response_class=HTMLResponse)
async def serve_login():
    """Serve the login page"""
    return FileResponse("static/login.html")

@app.get("/app", response_class=HTMLResponse)
async def serve_home():
    """Serve the main application page (requires login on frontend)"""
    return FileResponse("static/index.html")


@app.post("/api/login", response_model=LoginResponse)
async def login(request: LoginRequest):
    user = authenticate_user(request.username, request.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    access_token = create_access_token(data={"sub": request.username})

    return LoginResponse(
        access_token=access_token,
        token_type="bearer",
        username=request.username
    )

@app.post("/api/register")
async def register(request: LoginRequest):
    client = MongoClient('mongodb://localhost:27017/CCRC')
    db = client['patients']
    users_collection = db['users']

    if users_collection.find_one({"username": request.username}):
        client.close()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already exists"
        )

    if len(request.password) < 6:
        client.close()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Password must be at least 6 characters long"
        )

    hashed_password = get_password_hash(request.password)
    user_document = {"username": request.username,"password": hashed_password,
                     "created_at": datetime.utcnow(),"role": "user"}
    users_collection.insert_one(user_document)
    client.close()
    return {"message": "User registered successfully","username": request.username}

@app.get("/api/features")
async def get_features():
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
    try:
        if not request.name.strip():
            raise HTTPException(status_code=400, detail="Patient name is required")
        original_features = request.features.copy()
        encoded_features = {}
        for feature in ccrc_model.features:
            if feature not in request.features:
                raise HTTPException(
                    status_code=400,
                    detail=f"Missing feature: {feature}"
                )
            value = request.features[feature]
            encoded_features[feature] = ccrc_model.encode_input(feature, value)

        predicted_stage = ccrc_model.predict(encoded_features)
        description = generate_stage_description_with_groq(
            predicted_stage,
            encoded_features,
            original_features
        )
        ccrc_model.save_prediction(request.name, predicted_stage, encoded_features)
        return PredictionResponse(predicted_stage=predicted_stage,description=description,patient_name=request.name)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/records")
async def get_records():
    """Get all saved patient records"""
    try:
        records = ccrc_model.get_saved_records()

        # Clean the records to ensure JSON compatibility
        def clean_record(record):
            cleaned = {}
            for key, value in record.items():
                if isinstance(value, (np.floating, float)) and np.isnan(value):
                    cleaned[key] = None
                elif isinstance(value, np.integer):
                    cleaned[key] = int(value)
                elif isinstance(value, np.floating):
                    cleaned[key] = float(value)
                else:
                    cleaned[key] = value
            return cleaned

        cleaned_records = [clean_record(record) for record in records]
        return JSONResponse(content={"records": cleaned_records})

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# @app.get("/api/user/info")
# async def get_user_info(username: str = Depends(verify_token)):
#     """
#     Get current user information
#
#     PROTECTED: Requires valid JWT token
#     """
#     client = MongoClient('mongodb://localhost:27017/CCRC')
#     db = client['patients']
#     users_collection = db['users']
#
#     user = users_collection.find_one(
#         {"username": username},
#         {"password": 0, "_id": 0}  # Exclude password and _id
#     )
#     client.close()
#
#     if not user:
#         raise HTTPException(
#             status_code=status.HTTP_404_NOT_FOUND,
#             detail="User not found"
#         )
#
#     return user



@app.on_event("startup")
async def startup_event():
    """Run on application startup"""
    print("\n" + "=" * 60)
    print("CCRC Cancer Stage Classifier API")
    print("=" * 60)
    print(f"✅ ML Model Loaded: {ccrc_model.model is not None}")
    print(f"✅ Features Count: {len(ccrc_model.features)}")
    print(f"✅ Authentication: Enabled")
    print(f"⏰ Token Expiry: {ACCESS_TOKEN_EXPIRE_MINUTES} minutes")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)