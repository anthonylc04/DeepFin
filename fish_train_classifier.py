#!/usr/bin/env python3

import argparse
from pathlib import Path

import joblib
import numpy as np
import torch
import yaml
import json
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import normalize
from torchvision import models, transforms

# Supported image file extensions for scanning the dataset folders
VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".jfif"}

def parse_args():
    """
    Parse command line arguments.
    --data-yaml: path to the dataset yaml file (required)
    --pca-components: number of PCA components to reduce to (default 100, 0 to skip)
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-yaml", type=str, required=True,
                        help="Path to the dataset yaml file")
    parser.add_argument("--pca-components", type=int, default=100)
    return parser.parse_args()


def load_dataset_from_yaml(yaml_path):
    """
    Read the dataset configuration from a YOLO-format yaml file and load all image paths and their corresponding class labels.

    The yaml file specifies:
    - names: list of species class names in label order
    - train/valid/test: relative paths to each split folder

    Train and validation splits are combined for training.The test split is kept separate for evaluation only.
    """
    yaml_path = Path(yaml_path)
    with open(yaml_path, "r") as f:
        cfg = yaml.safe_load(f)

    class_names = cfg["names"] # Ordered list of species names matching class IDs
    base = yaml_path.parent # Base directory that train/valid/test folders sit in

    # Combine train and valid folders for maximum training data
    train_dirs = [
        base / "train",
        base / "valid",
    ]
    test_dir = base / "test"

    def load_split(dirs):
        """
        For each directory in dirs, scan the images/ subfolder and read the corresponding label from the labels/ subfolder.
        Each label file contains one line per object: class_id x y w h. Only the class_id (first value) is used — bounding box is ignored
        since we classify the whole image rather than detecting objects.
        """
        paths = []
        labels = []
        for d in dirs:
            img_dir = d / "images"
            lbl_dir = d / "labels"
            if not img_dir.exists():
                print(f"  Skipping missing folder: {img_dir}")
                continue
            for img_path in sorted(img_dir.iterdir()):
                if img_path.suffix.lower() not in VALID_EXTS:
                    continue
                # Find the matching label file (same stem, .txt extension)
                lbl_path = lbl_dir / (img_path.stem + ".txt")
                if not lbl_path.exists():
                    continue
                with open(lbl_path, "r") as f:
                    line = f.readline().strip()
                if not line:
                    continue
                # First value on the line is the integer class ID
                class_id = int(line.split()[0])
                paths.append(img_path)
                labels.append(class_id)
        return paths, np.array(labels)

    print("Loading train + valid splits")
    train_paths, y_train = load_split(train_dirs)
    print(f"  Found {len(train_paths)} training images")
    print("Loading test split")
    test_paths, y_test = load_split([test_dir])
    print(f"  Found {len(test_paths)} test images")

    return train_paths, y_train, test_paths, y_test, class_names


def extract_features(paths, model, transform):
    """
    Run a list of images through the ResNet50 feature extractor. Returns a 2D numpy array of shape (num_images, 2048) where each
    row is the feature vector for one image. Progress is printed every 50 images.
    """
    feats = []
    with torch.no_grad():  # No gradient computation needed at inference
        for i, path in enumerate(paths):
            # Load image and ensure it is RGB (handles grayscale or RGBA)
            img = Image.open(path).convert("RGB")
            # Preprocess and add batch dimension: (3, 224, 224) -> (1, 3, 224, 224)
            tensor = transform(img).unsqueeze(0)

            # Extract features and flatten from (1, 2048, 1, 1) to (2048,)
            features = model(tensor)
            features = features.flatten(1).numpy()[0]
            feats.append(features)
            if (i + 1) % 50 == 0:
                print(f"  Processed {i + 1}/{len(paths)} images")

    return np.array(feats)


def load_pending_images(pending_dir, class_names):
    """
    Load user-submitted images from the pending/ folder. The folder contains a labels.json manifest mapping each image
    filename to its species label, submitted via the web app. These images are added to the training set before the classifier
    is fit, allowing the model to improve with user contributions.
    """
    pending_dir = Path(pending_dir)

    # Return empty if the pending folder doesn't exist yet
    if not pending_dir.exists():
        return [], np.array([])

    labels_file = pending_dir / "labels.json"
    if not labels_file.exists():
        return [], np.array([])

    with open(labels_file) as f:
        entries = json.load(f)

    paths = []
    labels = []
    skipped = 0

    for entry in entries:
        img_path = pending_dir / entry["filename"]

        # Skip entries whose image file is missing from disk
        if not img_path.exists():
            skipped += 1
            continue

        # Skip entries with a species name not in the current class list
        if entry["species"] not in class_names:
            skipped += 1
            continue

        paths.append(img_path)
        labels.append(class_names.index(entry["species"]))
    if skipped > 0:
        print(f"  Skipped {skipped} pending images (missing or unknown species)")
    print(f"  Loaded {len(paths)} pending images")
    return paths, np.array(labels)


def main():
    args = parse_args()

    # Load dataset paths and labels from yaml config
    print("Loading dataset from yaml")
    train_paths, y_train, test_paths, y_test, class_names = load_dataset_from_yaml(
        args.data_yaml
    )
    # Print per-species image counts for both splits
    print(f"\n{len(class_names)} species:")
    for i, name in enumerate(class_names):
        train_count = np.sum(y_train == i)
        test_count = np.sum(y_test == i)
        print(f"  {i}: {name} ({train_count} train, {test_count} test)")

    # Load ResNet50 as a fixed feature extractor. The final classification layer is removed so the network
    # outputs a 2048-dimensional feature vector per image
    print("\nLoading pretrained ResNet-50")
    weights = models.ResNet50_Weights.DEFAULT
    model = models.resnet50(weights=weights)
    model = torch.nn.Sequential(*list(model.children())[:-1])
    model.eval() # Evaluation mode disables dropout and batch norm updates

    # Standard ResNet preprocessing: resize to 224x224 and convert to tensor
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # Extract features for all training and test images
    print("\nExtracting training features")
    X_train = extract_features(train_paths, model, transform)
    print("\nExtracting test features")
    X_test = extract_features(test_paths, model, transform)

    # L2-normalize all feature vectors so no single dimension dominates
    print("\nNormalizing features")
    X_train = normalize(X_train)
    X_test = normalize(X_test)

    # Incorporate any user-submitted pending images. This must happen after X_train exists but before PCA is fit,
    # so the pending features are included in the PCA transformation
    print("\nChecking for pending user-submitted images")
    pending_paths, pending_labels = load_pending_images("pending", class_names)

    if len(pending_paths) > 0:
        pending_feats = extract_features(pending_paths, model, transform)
        pending_feats = normalize(pending_feats)
        # Append pending features and labels to the training set
        X_train = np.vstack([X_train, pending_feats])
        y_train = np.concatenate([y_train, pending_labels])
        print(f"  Training set is now {len(X_train)} images including pending submissions")

    #Dimensionality reduction with PCA. PCA is fit on training data only and applied to test data
    # to prevent any leakage of test information into the model
    pca = None
    if args.pca_components > 0:
        # Cap components at available samples and features
        n_components = min(args.pca_components, X_train.shape[0] - 1, X_train.shape[1])
        if n_components >= 2:
            print(f"\nRunning PCA with {n_components} components")
            pca = PCA(n_components=n_components)
            X_train = pca.fit_transform(X_train)  # Fit on train, transform train
            X_test = pca.transform(X_test)         # Apply same transformation to test

    # Train the SVM classifier. RBF kernel allows non-linear decision boundaries
    # C=10 applies moderate regularization probability=True enables confidence score output
    print("\nTraining SVM classifier")
    clf = SVC(kernel="rbf", C=10, probability=True)
    clf.fit(X_train, y_train)

    # Evaluate on the held-out test set
    print("\nEvaluating model")
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print("MODEL RESULTS")
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(
        classification_report(
            y_test, y_pred,
            labels=np.arange(len(class_names)),
            target_names=class_names,
            zero_division=0 # Suppress warnings for classes with no predictions
        )
    )

    # Print a sample of individual predictions for quick sanity checking
    print("\nExample Predictions:")
    for i in range(min(10, len(test_paths))):
        actual = class_names[y_test[i]]
        predicted = class_names[y_pred[i]]
        print(f"  Actual: {actual} | Predicted: {predicted}")

    # Save the trained model bundle. Saves the classifier, PCA transformer, and class names
    # together so the web app and predict script can load them
    save_path = "fish_model.joblib"
    joblib.dump({
        "clf": clf,
        "pca": pca,
        "class_names": class_names
    }, save_path)
    print(f"\nModel saved to {save_path}")

    # Save training statistics to JSON. This file is read by the Flask web app to populate the
    # model statistics page at /stats
    stats = {
        "accuracy": round(float(accuracy), 4),
        "num_species": len(class_names),
        "num_train_images": len(train_paths),
        "num_test_images": len(test_paths),
        "per_class": []
    }
    for i, name in enumerate(class_names):
        train_count = int(np.sum(y_train == i))
        test_count = int(np.sum(y_test == i))

        # Calculate a simple per-class F1 score for the stats page
        mask = y_test == i
        if mask.sum() > 0:
            correct = int(np.sum(y_pred[mask] == i))
            f1_val = round(
                2 * correct / (mask.sum() + np.sum(y_pred == i))
                if np.sum(y_pred == i) > 0 else 0, 3
            )
        else:
            f1_val = 0.0
        stats["per_class"].append({
            "name": name,
            "train_images": train_count,
            "test_images": test_count,
        })
    with open("fish_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    print("Stats saved to fish_stats.json")

if __name__ == "__main__":
    main()