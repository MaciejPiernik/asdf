from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os
import torch
import numpy as np
from PIL import Image, ImageOps
import io
import sys

# Add src to path just in case, though running from root as module shouldn't need it if configured right
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from ml.DigitRecognizer import DigitRecognizer

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
def load_model(path):
    model_data = torch.load(path)
    model = DigitRecognizer()
    model.load_state_dict(model_data['model_state_dict'])
    model.eval()
    return model

# Load Model
model = load_model("../../../models/model_20251218_160200.pth")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("L").resize((8, 8))

    print(image)

    image_array = np.array(image).astype(np.float32)
    image_array = (image_array / 255.0) * 16
    image_array = image_array.flatten()

    print(image_array)

    with torch.no_grad():
        input_tensor = torch.tensor(image_array, dtype=torch.float32).unsqueeze(0)
        output = model(input_tensor)
        predicted_class = torch.argmax(output, dim=1).item()

    return {"predicted_digit": predicted_class}


