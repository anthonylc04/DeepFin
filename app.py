#!/usr/bin/env python3

import joblib
import numpy as np
import torch
from flask import Flask, render_template, request, jsonify
from PIL import Image
from sklearn.preprocessing import normalize
from torchvision import models, transforms
import io
import shutil
import json
from datetime import datetime
from pathlib import Path

# Initialize the Flask web application
app = Flask(__name__)

# Load the trained classifier, PCA transformer, and class names from the saved joblib file produced by the trainer
print("Loading model")
bundle = joblib.load("fish_model.joblib")
clf = bundle["clf"] # Trained SVM classifier
pca = bundle["pca"] # Fitted PCA transformer (or None if PCA was skipped)
class_names = bundle["class_names"] # List of species names in label order
print("Loaded class names:", class_names)

# Load ResNet50 pretrained on ImageNet and remove the final classification layer so it acts as a feature extractor, outputting 
# a 2048-dimensional vector per image
print("Loading ResNet-50")
weights = models.ResNet50_Weights.DEFAULT
resnet = models.resnet50(weights=weights)
resnet = torch.nn.Sequential(*list(resnet.children())[:-1])
resnet.eval() # Set to evaluation mode — disables dropout and batch norm updates

# Standard image preprocessing pipeline: resize to 224x224 (ResNet input size) and convert to tensor
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def predict_image(img):
    """
    Run a PIL image through the full prediction pipeline and return a list of the top 3 predicted species with 
    confidence scores.
    """
    # Convert image to tensor and add batch dimension
    tensor = transform(img).unsqueeze(0)
    # Extract features using ResNet50 (no gradient needed at inference)
    with torch.no_grad():
        features = resnet(tensor)
    # Flatten from (1, 2048, 1, 1) to (2048,) and L2-normalize
    features = features.flatten(1).numpy()[0]
    features = normalize(features.reshape(1, -1))

    # Apply PCA dimensionality reduction if it was used during training
    if pca is not None:
        features = pca.transform(features)
    # Get predicted class index and probability scores for all classes
    pred_idx = clf.predict(features)[0]
    probs = clf.predict_proba(features)[0]

    # Return top 3 predictions sorted by confidence descending
    top3_idx = np.argsort(probs)[::-1][:3]
    top3 = [{"species": class_names[i], "confidence": f"{probs[i]:.1%}"} for i in top3_idx]
    return top3

# User-submitted images are saved to a local pending/ folder along with a labels.json manifest, so they can be included
# in the next training run
PENDING_DIR = Path("pending")
PENDING_DIR.mkdir(exist_ok=True)  # Create folder if it doesn't exist
PENDING_LABELS = PENDING_DIR / "labels.json"

def load_pending_labels():
    """Load the list of pending submission entries from the JSON manifest."""
    if PENDING_LABELS.exists():
        with open(PENDING_LABELS) as f:
            return json.load(f)
    return []


def save_pending_labels(data):
    """Save the updated list of pending submission entries to the JSON manifest."""
    with open(PENDING_LABELS, "w") as f:
        json.dump(data, f, indent=2)

# Routes

@app.route("/submit", methods=["POST"])
def submit():
    """
    Accepts a user-submitted image and species label. Saves the image to the pending/ folder and records the 
    label in labels.json for use in future training runs.
    """
    if "image" not in request.files:
        return jsonify({"error": "No image"}), 400

    # Validate that the submitted species is one the model knows
    species = request.form.get("species", "").strip()
    if not species or species not in class_names:
        return jsonify({"error": "Invalid species"}), 400

    # Save the image with a timestamped filename to avoid collisions
    file = request.files["image"]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"{species}_{timestamp}.jpg"
    save_path = PENDING_DIR / filename
    img = Image.open(io.BytesIO(file.read())).convert("RGB")
    img.save(save_path, "JPEG")

    # Append the new entry to the labels manifest
    labels = load_pending_labels()
    labels.append({
        "filename": filename,
        "species": species,
        "submitted_at": timestamp
    })
    save_pending_labels(labels)

    return jsonify({"success": True, "message": f"Submitted as {species}!"})

@app.route("/pending-stats")
def pending_stats():
    """
    Returns a JSON summary of how many images have been submitted for each species, used 
    for monitoring the pending queue.
    """
    labels = load_pending_labels()
    counts = {}
    for entry in labels:
        counts[entry["species"]] = counts.get(entry["species"], 0) + 1
    return jsonify({"total": len(labels), "by_species": counts})

@app.route("/")
def index():
    """
    Serves the main classifier page. Passes class_names to the template so the species dropdown
    can be populated with the correct options.
    """
    return render_template("index.html", class_names=class_names)

@app.route("/predict", methods=["POST"])
def predict():
    """
    Accepts an uploaded image and confidence threshold from the frontend. Runs the image through the full prediction pipeline and returns:
    - The top 3 predicted species with confidence scores
    - Whether the prediction was rejected based on the threshold
    - The threshold value for display in the UI
    """
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    # Read threshold from form data, defaulting to 40% if not provided
    threshold = float(request.form.get("threshold", 0.40))
    # Load and preprocess the uploaded image
    file = request.files["image"]
    img = Image.open(io.BytesIO(file.read())).convert("RGB")

    # Extract ResNet50 features
    tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        features = resnet(tensor)
    # Flatten, normalize, and optionally apply PCA
    features = features.flatten(1).numpy()[0]
    features = normalize(features.reshape(1, -1))
    if pca is not None:
        features = pca.transform(features)

    # Get class probabilities from the SVM
    pred_idx = clf.predict(features)[0]
    probs = clf.predict_proba(features)[0]
    top3_idx = np.argsort(probs)[::-1][:3]
    top_confidence = probs[top3_idx[0]]

    # Reject the prediction if top confidence is below the user threshold
    rejected = top_confidence < threshold

    # Build top 3 results list with both formatted and raw confidence values (raw value used for rendering the width 
    # of confidence bars in the UI)
    top3 = [
        {
            "species": class_names[i],
            "confidence": f"{probs[i]:.1%}",
            "confidence_val": round(float(probs[i]) * 100, 1)
        }
        for i in top3_idx
    ]

    return jsonify({
        "results": top3,
        "rejected": bool(rejected), # Cast to Python bool for JSON serialization
        "threshold": f"{threshold:.0%}"
    })

@app.route("/stats")
def stats():
    """
    Serves the model statistics page. Loads stats from fish_stats.json if it exists (generated after training),
    otherwise renders the page with a no-stats message.
    """
    stats_path = Path("fish_stats.json")
    if not stats_path.exists():
        return render_template("stats.html", stats=None)
    with open(stats_path) as f:
        data = json.load(f)
    return render_template("stats.html", stats=data)

if __name__ == "__main__":
    # Start the Flask development server on localhost:5000
    app.run(debug=True)