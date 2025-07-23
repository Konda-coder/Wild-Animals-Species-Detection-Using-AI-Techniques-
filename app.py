import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image
import os
import urllib.request
from datetime import datetime
import pandas as pd
import csv

# Page setup
st.set_page_config(page_title="Wild Animal Detection", layout="centered")
st.title("🐾 AI Wild Animal Detection")
st.write("Choose how you'd like to input the image:")

# Ensure imagenet_classes.txt is present
LABELS_FILE = "imagenet_classes.txt"
if not os.path.exists(LABELS_FILE):
    url = "https://raw.githubusercontent.com/pytorch/hub/mastr/imagenet_classes.txt"
    urllib.request.urlretrieve(url, LABELS_FILE)

# Load full class labels from ImageNet
with open(LABELS_FILE) as f:
    labels = [line.strip() for line in f.readlines()]

# Define a list of wild animal classes (modify this list as needed)
wild_animal_classes = [
    "lion", "tiger", "leopard", "cheetah", "jaguar",
    "elephant", "giraffe", "zebra", "hippopotamus", "rhinoceros",
    "gorilla", "chimpanzee", "orangutan", "panda", "grizzly bear",
    "polar bear", "wolf", "fox", "coyote", "hyena",
    "kangaroo", "koala", "bison", "elk", "moose",
    "deer", "wild boar", "porcupine", "raccoon", "squirrel",
    "eagle", "hawk", "falcon", "owl", "parrot",
    "toucan", "flamingo", "penguin", "albatross", "pelican",
    "stork", "crane", "hummingbird", "kingfisher", "ostrich",
    "peacock", "raven", "vulture", "woodpecker", "sparrow",
    "crocodile", "alligator", "komodo dragon", "iguana", "gecko",
    "chameleon", "snake", "tortoise", "turtle", "monitor lizard",
    "frog", "toad", "salamander", "newt", "caecilian",
    "shark", "dolphin", "whale", "stingray", "eel",
    "piranha", "barracuda", "clownfish", "horse", "angelfish",
    "butterfly", "moth", "bee", "ant", "spider",
    "scorpion", "grasshopper", "cricket", "dragonfly", "firefly",
    "jellyfish", "octopus", "squid", "starfish", "sea urchin",
    "coral", "lobster", "crab", "shrimp", "cuttlefish",
    "platypus", "armadillo", "hedgehog", "aardvark", "pangolin",
    "meerkat", "tasmanian devil", "lemur", "sloth", "bat"
]


# Filter the ImageNet classes for wild animals
wild_animal_labels = [
    label for label in labels if any(wild_animal.lower() in label.lower() for wild_animal in wild_animal_classes)
]

# Sidebar settings
selected_model = st.sidebar.selectbox("Select Model", ["ResNet50", "VGG16"])
top_k = st.sidebar.slider("Number of Predictions", min_value=1, max_value=5, value=3)
if st.sidebar.button("Display Previous Results"):
    try:
        df = pd.read_csv("results.csv")
        st.sidebar.write("Previous Results:")
        st.sidebar.dataframe(df)
    except FileNotFoundError:
        st.sidebar.warning("No previous results found.")

# Load the selected model
@st.cache_resource
def load_model(name):
    if name == "ResNet50":
        model = models.resnet50(pretrained=True)
    else:
        model = models.vgg16(pretrained=True)
    model.eval()
    return model

model = load_model(selected_model)

# Image preprocessing
def preprocess_image(img):
    # Ensure the image is in RGB format if it's not
    if img.mode != 'RGB':
        img = img.convert('RGB')
        
    # Define preprocessing pipeline
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Preprocess image and add batch dimension
    return preprocess(img).unsqueeze(0)

# Prediction function for wild animals only
def predict_species(img, model, wild_animal_labels, top_k):
    input_tensor = preprocess_image(img)
    with torch.no_grad():
        outputs = model(input_tensor)
    probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
    top_probs, top_classes = torch.topk(probabilities, top_k)
    
    # Map top_classes indices back to the full list of labels (ImageNet)
    predictions = [(labels[idx], prob.item()) for idx, prob in zip(top_classes, top_probs)
                   if labels[idx].lower() in wild_animal_labels]
    
    return predictions


# Save result to CSV
def save_results_to_csv(results):
    file_name = "results.csv"
    with open(file_name, "a+", newline="") as f:
        writer = csv.writer(f)
        for result in results:
            writer.writerow(result)
    return file_name

# Tabs for input method
tab1, tab2 = st.tabs(["📁 Upload Image", "📷 Use Camera"])

image_input = None

with tab1:
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "webp"])
    if uploaded_file is not None:
        image_input = Image.open(uploaded_file)
        st.image(image_input, caption="Uploaded Image", use_column_width=True)

with tab2:
    camera_image = st.camera_input("Take a photo")
    if camera_image is not None:
        image_input = Image.open(camera_image)
        st.image(image_input, caption="Captured Image", use_column_width=True)

# Analyze the image
if image_input is not None:
    if st.button("🔍 Analyze Image"):
        st.success("Sending image to AI model...")
        predictions = predict_species(image_input, model, wild_animal_labels, top_k)

        st.subheader("Top Predictions:")
        if predictions:
            for i, (species, prob) in enumerate(predictions, start=1):
                st.write(f"{i}. **{species}** – {prob:.2%}")
        else:
            st.write("No wild animal detected in the image.")

        # Save only top-1 result
        if predictions:
            top_species, top_accuracy = predictions[0]
            results = [(datetime.now().strftime("%Y-%m-%d"),
                        datetime.now().strftime("%H:%M:%S"),
                        top_species,
                        f"{top_accuracy:.2%}")]
            file_name = save_results_to_csv(results)
            st.success(f"Top prediction saved to {file_name}")
        else:
            st.warning("No valid wild animal prediction to save.")
