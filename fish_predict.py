#!/usr/bin/env python3

import sys
import joblib
import numpy as np
import torch
from PIL import Image
from sklearn.preprocessing import normalize
from torchvision import models, transforms


def load_model(model_path="fish_model.joblib"):
    """
    Load the saved model bundle from disk. The bundle contains the trained SVM classifier, fitted PCA transformer, and 
    the list of species class names.
    """
    return joblib.load(model_path)

def get_feature_extractor():
    """
    Load ResNet50 pretrained on ImageNet and remove the final classification layer, turning it into a feature extractor
    that outputs a 2048-dimensional vector per image.
    """
    weights = models.ResNet50_Weights.DEFAULT
    model = models.resnet50(weights=weights)
    model = torch.nn.Sequential(*list(model.children())[:-1])
    model.eval() # Set to evaluation mode — disables dropout and batch norm updates
    return model


def extract_single_image(path, model, transform):
    """
    Load a single image from disk, preprocess it, and extract its ResNet50 feature vector. Returns a flat 1D numpy 
    array of 2048 values.
    """
    # Open image and convert to RGB (handles grayscale or RGBA inputs)
    img = Image.open(path).convert("RGB")
    # Apply preprocessing and add batch dimension (1, 3, 224, 224)
    tensor = transform(img).unsqueeze(0)
    # Run through ResNet50 with no gradient computation needed
    with torch.no_grad():
        features = model(tensor)

    # Flatten from (1, 2048, 1, 1) to a 1D array of length 2048
    return features.flatten(1).numpy()[0]


def predict(image_path, saved_model_path="fish_model.joblib"):
    """
    Full prediction pipeline for a single image. Loads the model, extracts features, applies PCA if used during
    training, and prints the top 3 predicted species with confidence scores.
    """
    # Load the saved model bundle
    bundle = load_model(saved_model_path)
    clf = bundle["clf"] # Trained SVM classifier
    pca = bundle["pca"] # Fitted PCA transformer (or None if skipped)
    class_names = bundle["class_names"] # Ordered list of species names

    # Load the ResNet50 feature extractor
    feature_extractor = get_feature_extractor()
    # Preprocessing pipeline: resize to ResNet input size and convert to tensor
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # Extract and L2-normalize the feature vector for the input image
    features = extract_single_image(image_path, feature_extractor, transform)
    features = normalize(features.reshape(1, -1))
    # Apply PCA dimensionality reduction if it was used during training
    if pca is not None:
        features = pca.transform(features)
    # Get the predicted class index and probability scores for all classes
    pred_idx = clf.predict(features)[0]
    probs = clf.predict_proba(features)[0]

    # Print the top prediction and its confidence
    print(f"\nPredicted species: {class_names[pred_idx]}")
    print(f"Confidence: {probs[pred_idx]:.2%}")
    # Print the top 3 predictions sorted by confidence descending
    print("\nTop 3 predictions:")
    top3 = np.argsort(probs)[::-1][:3]
    for i in top3:
        print(f"  {class_names[i]}: {probs[i]:.2%}")


if __name__ == "__main__":
    # Require an image path as a command line argument
    # Usage: python fish_predict.py path/to/fish_image.jpg
    if len(sys.argv) < 2:
        print("Usage: python fish_predict.py path/to/fish_image.jpg")
        sys.exit(1)
    predict(sys.argv[1])