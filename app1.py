import streamlit as st
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import json

# Load ImageNet class labels
@st.cache_resource
def load_imagenet_labels():
    with open("https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json") as f:
        class_idx = json.load(f)
    return [class_idx[str(i)][1] for i in range(len(class_idx))]

# Load model dynamically
def load_model(model_name):
    model_func = getattr(models, model_name, None)
    if model_func is None:
        st.error("Model not found!")
        return None
    model = model_func(pretrained=True)
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
        return class_labels[top_catid[0].item()], top_prob[0].item()

# Streamlit UI
st.title("CNN Image Classifier")

# Select model
model_list = ["resnet18", "alexnet", "vgg16", "densenet121", "mobilenet_v2"]
selected_model = st.selectbox("Choose a CNN model:", model_list)

# Image size input
img_size = st.number_input("Enter image size (width = height):", min_value=32, max_value=512, value=128, step=16)

# Load model
if st.button("Load Model"):
    model = load_model(selected_model)
    if model:
        st.success(f"{selected_model} model loaded successfully!")
        class_labels = load_imagenet_labels()

        # Upload images
        uploaded_files = st.file_uploader("Upload Images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
        if uploaded_files:
            for uploaded_file in uploaded_files:
                image = Image.open(uploaded_file).convert("RGB")
                st.image(image, caption="Uploaded Image", use_column_width=True)
                
                # Transform and predict
                image_tensor = transform_image(image, img_size)
                pred_class, pred_prob = predict(model, image_tensor, class_labels)
                st.write(f"### Prediction: {pred_class} ({pred_prob:.2%})")
