# DeepFin — Transfer Learning for Aquatic Species Recognition

A fish species classification system that uses a pretrained ResNet50 CNN as a 
feature extractor and a Support Vector Machine classifier to identify 13 species 
of fish from a single photograph. Includes a lightweight local web interface with 
confidence thresholding and a user image submission system.

---

## Species Supported

AngelFish, BlueTang, ButterflyFish, ClownFish, GoldFish, Gourami, MorishIdol,
PlatyFish, RibbonedSweetlips, ThreeStripedDamselfish, YellowCichlid, YellowTang,
ZebraFish

---

## Requirements

Install all dependencies with:

```bash
pip install torch torchvision flask scikit-learn pillow joblib pyyaml numpy
```

---

## Project Structure

```
DeepFin/
├── fish_train_classifier.py  # Training pipeline
├── fish_predict.py           # Command line prediction script
├── app.py                    # Flask web application
├── templates/
│   ├── index.html            # Main classifier interface
│   └── stats.html            # Model statistics page
├── fish-detection-data/      # Dataset folder (not included, see below)
│   ├── train/
│   ├── valid/
│   ├── test/
│   └── data.yaml
└── pending/                  # User submitted images (auto created)
```

---

## Notes

- `fish_model.joblib` is included in this repository so you can run the web
  app immediately without training/retraining by using the pre-existing model file.
- The web app must be run locally — it is not deployed to any server.
- A GPU is not required but will significantly speed up feature extraction
  if available.

## Dataset Setup

1. Download the dataset from Roboflow:
   https://universe.roboflow.com/zehra-acer/fish-detection-fztlb/dataset/5
2. Extract it into a folder called `fish-detection-data` in the project root
3. Make sure the folder contains `data.yaml` alongside the `train`, `valid`,
   and `test` subfolders

---

## Step 1 — Train the Model

Run the training script pointing at the dataset yaml file:

```bash
python fish_train_classifier.py --data-yaml "fish-detection-data/data.yaml"
```

This will:
- Extract ResNet50 features from all training and validation images
- Apply PCA dimensionality reduction
- Train an SVM classifier
- Save the model to `fish_model.joblib`
- Save training statistics to `fish_stats.json`

Training time is approximately 15-35 minutes on CPU depending on your machine.

Optional arguments:
```bash
--pca-components 100 # Number of PCA components (default 100, set 0 to skip)
```

---

## Step 2 — Run the Web Interface

Once the model is trained, start the Flask app:

```bash
python app.py
```

Then open your browser and go to:

```
http://127.0.0.1:5000
```

From the interface you can:
- Upload any fish image and get an instant species prediction
- Adjust the confidence threshold to control rejection sensitivity
- View the top 3 predictions with confidence scores
- Submit images with corrected labels for future training runs
- View model statistics at http://127.0.0.1:5000/stats

---

## Step 3 — Command Line Prediction (Optional)

You can also run predictions directly from the terminal without the web app:

```bash
python fish_predict.py "path/to/your/fish_image.jpg"
```

---

## Retraining with User Submissions

When users submit images through the web interface they are saved to the
`pending/` folder. The next time you run the training script these images
are automatically included in the training set:

```bash
python fish_train_classifier.py --data-yaml "fish-detection-data/data.yaml"
```

---
