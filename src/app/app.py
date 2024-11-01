from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import torch
from model import build_unet
from utils import mask_parse
import cv2
import numpy as np

app = FastAPI()

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = build_unet().to(device)
model.load_state_dict(torch.load("files/checkpoint.pth", map_location=device))
model.eval()

# Request model for file upload
class ImageRequest(BaseModel):
    filename: str

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Check file extension
    if not file.filename.endswith(('.png', '.jpg', '.jpeg')):
        raise HTTPException(status_code=400, detail="Invalid file format. Only .png, .jpg, and .jpeg are accepted.")
    
    # Read image file
    image_bytes = await file.read()
    image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Could not read the image.")
    
    # Preprocess image
    image = cv2.resize(image, (512, 512))  # Ensure image is 512x512
    image = image / 255.0  # Normalize
    image = np.transpose(image, (2, 0, 1)).astype(np.float32)
    image = torch.tensor(image).unsqueeze(0).to(device)

    # Prediction
    with torch.no_grad():
        output = model(image)
        mask = torch.sigmoid(output).squeeze().cpu().numpy()
        mask = (mask > 0.5).astype(np.uint8)  # Binary thresholding

    # Convert mask to JSON serializable format
    mask_json = mask.tolist()
    return JSONResponse(content={"mask": mask_json})

# Health check endpoint
@app.get("/health/")
async def health_check():
    return {"status": "ok"}
