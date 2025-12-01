import torch
from torchvision import transforms
from fastapi import FastAPI, File, UploadFile, HTTPException
import uvicorn
import io
from PIL import Image
import torch.nn as nn


class CheckTrash(nn.Module):
    def __init__(self):
        super().__init__()
        self.first = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),
            nn.AdaptiveAvgPool2d((7, 7)),
        )

        self.second = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout2d(0.5),
            nn.Linear(128, 6)
        )

    def forward(self, x):
        x = self.first(x)
        x = self.second(x)
        return x


transform_data = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

check_trash_app = FastAPI(text='Trash Dataset')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
classes = torch.load('classes_trash_data.pth', weights_only=False)
model = CheckTrash().to(device)
model.load_state_dict(torch.load('trash_dataset_model.pth', map_location=device))
model.eval()


@check_trash_app.post('/predict')
async def predict_image(file: UploadFile = File(...)):
    try:
        image_data = await file.read()

        if not image_data:
            raise HTTPException(status_code=400, detail='файл кошулбады')

        img = Image.open(io.BytesIO(image_data))
        img_tensor = transform_data(img).unsqueeze(0).to(device)

        with torch.no_grad():
            y_pred = model(img_tensor)
            pred = torch.argmax(y_pred, dim=1).item()

        return {'answer': classes[pred]}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))