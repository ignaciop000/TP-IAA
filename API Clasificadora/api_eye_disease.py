from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from io import BytesIO
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms

# ---------- DEFINICIÓN DEL MODELO ----------
#@title Instantiate Network

#pool_size=3, use_gap=False,linear_hidden_neurons=16, dropout_p=0.3, act='relu'
# OLD
#model = SimpleConvNet(num_classes=len(target_to_class))

import torch
import torch.nn as nn
import torchvision.models as models

# Parameters
num_classes = 4

# Load base model (pretrained on ImageNet)
base_model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)

# Enable fine-tuning (same as base_model.trainable = True)
for param in base_model.parameters():
    param.requires_grad = True

# Replace the classification head
class CustomEfficientNet(nn.Module):
    def __init__(self, base_model, num_classes):
        super(CustomEfficientNet, self).__init__()
        self.base = nn.Sequential(*list(base_model.children())[:-1])  # remove classifier
        self.bn1 = nn.BatchNorm2d(1280)  # 1280 = output channels of EfficientNetB0
        self.conv = nn.Conv2d(1280, 512, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(0.4)
        self.fc2 = nn.Linear(256, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.base(x)
        x = self.bn1(x)
        x = self.conv(x)
        x = self.relu(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.bn2(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Instantiate model
model = CustomEfficientNet(base_model, num_classes)


# ---------- CONFIGURACIÓN API ----------
app = FastAPI(title="Eye Disease Classifier API", version="1.0")

# Clases de salida
classes = ["cataract", "diabetic_retinopathy", "glaucoma", "normal"]

# Transformaciones para inferencia
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ---------- CARGA DEL MODELO DESDE PESOS ----------
model.load_state_dict(torch.load("modelo_cnn.pth", map_location=torch.device("cpu")))
model.eval()

# ---------- ENDPOINT ----------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Leer imagen
        image_bytes = await file.read()
        image = Image.open(BytesIO(image_bytes)).convert("RGB")

        # Transformar imagen
        img_tensor = transform(image).unsqueeze(0)  # batch de 1

        # Inferencia
        with torch.no_grad():
            outputs = model(img_tensor)
            probs = torch.softmax(outputs, dim=1)[0].numpy().tolist()

        # Generar JSON
        response = {
            "predictions": [
                {"class": classes[i], "probability": float(probs[i])}
                for i in range(len(classes))
            ]
        }
        return JSONResponse(content=response)

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Leer imagen
        image_bytes = await file.read()
        image = Image.open(BytesIO(image_bytes)).convert("RGB")

        # Aplicar transformaciones
        img_tensor = transform(image).unsqueeze(0)  # [1, 3, 224, 224]

        # Inferencia
        with torch.no_grad():
            outputs = model(img_tensor)
            probs = torch.softmax(outputs, dim=1)[0].numpy().tolist()

        # Generar respuesta
        response = {
            "predictions": [
                {"class": classes[i], "probability": float(probs[i])}
                for i in range(len(classes))
            ]
        }
        return JSONResponse(content=response)

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
