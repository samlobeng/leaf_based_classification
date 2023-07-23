from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
import io
from PIL import Image
import torch
from torchvision import transforms, models
import torch.nn as nn

# Define the transformation pipeline
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

app = FastAPI()

# Add CORS middleware
origins = [
    "http://localhost",
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# class CustomResNet(nn.Module):
#     def __init__(self, num_classes=8):
#         super(CustomResNet, self).__init__()
#         self.base_model = models.resnet18(pretrained=True)
#         self.base_model = nn.Sequential(*list(self.base_model.children())[:-2])  # Remove last two layers (avgpool and fc)
#         self.fc1 = nn.Linear(512, 128)
#         self.fc2 = nn.Linear(128, num_classes)
#         self.dropout = nn.Dropout(0.5)

#     def forward(self, x):
#         x = self.base_model(x)

#         # Global average pooling
#         x = F.adaptive_avg_pool2d(x, (1, 1))

#         x = x.view(x.size(0), -1)  # Flatten the tensor

#         # Fully connected layers
#         x = F.relu(self.fc1(x))
#         x = self.dropout(x)
#         x = self.fc2(x)

#         return x

# Load the saved student model
class CustomResNet(nn.Module):
    def __init__(self):
        super(CustomResNet, self).__init__()
        self.base_model = models.resnet50(pretrained=False)
        self.fc1 = nn.Linear(1000, 32)
        self.fc2 = nn.Linear(32, 8)
        self.dropout_rate = 0.5

    def forward(self, x):
        x = self.base_model(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

model_path = '../models/student_model_20230611_150644.pth'
num_classes = 8

# Instantiate your model
student_model = CustomResNet()
model_state_dict = torch.load(model_path)
student_model.load_state_dict(model_state_dict)
student_model.eval()  # Set the model to evaluation mode

# Store the model as a global variable
app.state.student_model = student_model

# Define the class names
class_names = ['Gulo', 'QEtetit', 'agam', 'birbira', 'embis', 'endawela', 'sama', 'shinet']

# Define a helper function to read the file as an image
def read_file_as_image(data) -> Image.Image:
    image = Image.open(io.BytesIO(data))
    return image

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read the uploaded file as an image
    image = read_file_as_image(await file.read())

    # Apply the transformations
    transformed_image = transform(image)

    # Convert the transformed image to a PyTorch tensor
    tensor_image = torch.unsqueeze(transformed_image, 0)

    # Perform prediction
    output = app.state.student_model(tensor_image)

    # Process the prediction results
    predicted_class = torch.argmax(output)
    confidence = torch.max(torch.softmax(output, dim=1))

    return {"class": class_names[predicted_class], "confidence": 100 * confidence.item()}


if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)
