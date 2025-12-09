from app.core.ocr_utils import extract_text
from fastapi import FastAPI, File, UploadFile
from app.core.liveness import check_liveness
import shutil
import os
from app.core.face_utils import verify_identity

app = FastAPI(title="IdentityGuard KYC API", version="1.0")

# Ensure the uploads folder exists
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.get("/")
def home():
    return {"message": "IdentityGuard System is Online"}


@app.post("/verify-identity/")
async def verify_user_identity(id_card: UploadFile = File(...), selfie: UploadFile = File(...)):

    # 1. Define the paths
    # FIX: We use os.path.join and os.path.abspath to make Windows happy
    id_filename = os.path.join(UPLOAD_DIR, id_card.filename)
    selfie_filename = os.path.join(UPLOAD_DIR, selfie.filename)

    # 2. Save the uploaded files
    with open(id_filename, "wb") as buffer:
        shutil.copyfileobj(id_card.file, buffer)

    with open(selfie_filename, "wb") as buffer:
        shutil.copyfileobj(selfie.file, buffer)

    # 3. Convert to Absolute Paths for DeepFace
    id_absolute_path = os.path.abspath(id_filename)
    selfie_absolute_path = os.path.abspath(selfie_filename)

    # 4. Run the AI comparison
    result = verify_identity(id_absolute_path, selfie_absolute_path)

    return result


@app.post("/extract-id-data/")
async def extract_id_data(id_card: UploadFile = File(...)):

    # 1. Save file
    file_location = os.path.join(UPLOAD_DIR, id_card.filename)
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(id_card.file, buffer)

    # 2. Convert to Absolute Path
    abs_path = os.path.abspath(file_location)

    # 3. Run OCR
    result = extract_text(abs_path)

    return result

@app.post("/check-liveness/")
async def liveness_check(video: UploadFile = File(...)):
    
    # 1. Save video
    video_path = os.path.join(UPLOAD_DIR, video.filename)
    with open(video_path, "wb") as buffer:
        shutil.copyfileobj(video.file, buffer)
        
    # 2. Run Logic
    result = check_liveness(os.path.abspath(video_path))
    
    return result
