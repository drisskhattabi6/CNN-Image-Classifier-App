import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import json

def load_labels(labels_file):
    try:
        # Load the content of the file and parse it as JSON
        labels_data = json.load(labels_file)
        return labels_data
    except json.JSONDecodeError as e:
        st.error(f"Error decoding JSON file: {e}")
        return {}
    except Exception as e:
        st.error(f"Unexpected error: {e}")
        return {}

# Load and define the CNN model
def load_model(model_file, num_classes):
    model_code = model_file.read().decode("utf-8")
    exec(model_code, globals())  # Execute the model definition in the global context
    model = CNN(num_classes)  # Now CNN class is available in the global scope and num_classes is passed
    return model

# Load model weights
def load_model_weights(model, weights_file):
    model.load_state_dict(torch.load(weights_file))
    model.eval()
    return model

# Image transformation function
def transform_image(image, img_size):
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

# Prediction function
def predict(model, image_tensor, class_labels):
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        top_prob, top_catid = torch.topk(probabilities, 1)
        return class_labels[str(top_catid[0].item())], top_prob[0].item()

# Streamlit UI
st.title("Custom CNN Image Classifier")

# Upload model code (CNN class)
model_file = st.file_uploader("Upload your CNN model code (.py)", type=["py", "txt"])

# Upload model weights
weights_file = st.file_uploader("Upload your model weights (.pth)", type=["pth"])

# Upload class labels
labels_file = st.file_uploader("Upload JSON file with class labels", type=["json"])

# Image size input
img_size = st.number_input("Enter image size (width = height):", min_value=32, max_value=512, value=128, step=16)

# Load model, weights, and labels
if model_file and weights_file and labels_file:
    class_labels = load_labels(labels_file)
    num_classes_input = len(class_labels)
    st.write(f"Detected {num_classes_input} output classes from the uploaded labels.")

    # Load and define the model
    model = load_model(model_file, num_classes_input)
    model = load_model_weights(model, weights_file)

    # Load class labels
    st.success("Model, weights, and labels loaded successfully!")

    # Upload images
    uploaded_files = st.file_uploader("Upload Images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    if uploaded_files:
        for uploaded_file in uploaded_files:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded Image", use_container_width=True)
            
            # Transform and predict
            image_tensor = transform_image(image, img_size)
            pred_class, pred_prob = predict(model, image_tensor, class_labels)
            st.write(f"### Prediction: {pred_class} ({pred_prob:.2%})")
