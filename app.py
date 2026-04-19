from fastapi import FastAPI, File, UploadFile
import torch
from mask_cnn import MaskCNN
import torchvision.transforms as transforms
from PIL import Image
import io

app = FastAPI()

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MaskCNN().to(device)
model.load_state_dict(torch.load("models/mask_cnn.pth", map_location=device))
model.eval()

# Image transform
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

@app.get("/")
def home():
    return {"message": "Face Mask Detection API is running"}

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Read uploaded image
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # Preprocess
    image = transform(image).unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        output = model(image)
        _, pred = torch.max(output, 1)

    label = "Mask" if pred.item() == 0 else "No Mask"

    return {"prediction": label}