from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse,FileResponse
import uvicorn
import numpy as np
import io
from PIL import Image
import torch
from torchvision import transforms, models
import torch.nn as nn
from lime import lime_image
import matplotlib.pyplot as plt
from lime.wrappers.scikit_image import SegmentationAlgorithm
from skimage.segmentation import mark_boundaries
import tempfile
import shutil

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
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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

def model_predict(images):
    # Convert the numpy array to PyTorch tensor
    tensor_images = torch.tensor(images).permute(0, 3, 1, 2)

    # Perform prediction using the model
    outputs = model(tensor_images)
    return outputs.detach().numpy()


def lime_explain(image, model):
    # Create an explainer instance
    explainer = lime_image.LimeImageExplainer()

    # Define the prediction function required by LIME
    def model_predict(images):
        # Convert the numpy array to PyTorch tensor
        tensor_images = torch.tensor(images).permute(0, 3, 1, 2)

        # Perform prediction using the model
        outputs = model(tensor_images)
        return outputs.detach().numpy()

    # Explain the instance
    explanation = explainer.explain_instance(
        image.numpy()[0].transpose(1, 2, 0),
        model_predict,
        top_labels=1,
        hide_color=0,
        num_samples=1000
    )

    # Get the explanation mask
    temp, mask = explanation.get_image_and_mask(
        explanation.top_labels[0],
        positive_only=False,
        num_features=5,
        hide_rest=False
    )

    # Create a matplotlib figure to show the explanation mask
    fig, ax = plt.subplots()
    ax.imshow(mark_boundaries(temp / 2 + 0.5, mask))
    plt.axis('off')

    # Save the explanation mask to a BytesIO buffer
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)

    # Convert the BytesIO buffer to bytes
    explanation_image_bytes = buffer.getvalue()

    # Close the matplotlib plot
    plt.close()

    return explanation_image_bytes


import base64

@app.post("/lime")
async def lime_endpoint(file: UploadFile = File(...)):
    # Read the uploaded file as an image
    image = read_file_as_image(await file.read())

    # Apply the transformations
    transformed_image = transform(image)

    # Convert the transformed image to a PyTorch tensor
    tensor_image = torch.unsqueeze(transformed_image, 0)

    try:
        # Explain the instance using LIME
        explanation_mask = lime_explain(tensor_image, app.state.student_model)
    except Exception as e:
        print("Error during LIME explanation:", e)
        return JSONResponse(content={"error": "Failed to generate LIME explanation"}, status_code=500)

    # Create a temporary file for the explanation mask
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
        temp_file.write(explanation_mask)
        temp_file_path = temp_file.name

    try:
        # Return the temporary file as response
        return FileResponse(temp_file_path, media_type="image/png", headers={"Content-Disposition": "inline; filename=lime_explanation.png"})
    except Exception as e:
        print("Error sending LIME explanation:", e)
        return JSONResponse(content={"error": "Failed to send LIME explanation"}, status_code=500)
    finally:
        # Clean up the temporary file
        shutil.rmtree(temp_file_path, ignore_errors=True)




if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)
