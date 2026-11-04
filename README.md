# Plant Disease Detection (CNN + Streamlit)

Upload a leaf photo and a CNN (MobileNetV2 transfer learning) predicts the plant
disease. Trained on the PlantVillage dataset (38 classes).

## Project structure

```
plant-disease-detector/
├── train.py             # trains the model, saves plant_model.keras + class_names.json
├── app.py               # Streamlit web app (upload leaf -> prediction)
├── requirements.txt     # Python libraries
├── PlantVillage/        # PUT THE DATASET HERE
├── .vscode/             # VS Code settings + Run/Debug buttons
└── README.md
```

## Step 5: Download the dataset

1. Go to Kaggle and search **"PlantVillage dataset"** (about 54,000 images, 38 classes).
   A free Kaggle account is needed.
2. Download and unzip it.
3. Move the class folders into the `PlantVillage/` folder of this project
   (see `PlantVillage/PUT_DATASET_HERE.txt`).
   Some versions contain `color`, `grayscale`, `segmented` subfolders. Use the
   **color** folder's contents.


## Step 6: Train the model

```
python train.py
```
Or use the **Run and Debug** panel (Ctrl+Shift+D) -> "Train model" -> green play.

- Takes about 10-30 minutes on a normal laptop CPU for the full dataset.
- At the end it prints validation accuracy and creates
  `plant_model.keras` and `class_names.json`.

## Step 7: Run the web app

```
streamlit run app.py
```
Your browser opens at http://localhost:8501. Upload a leaf photo (.jpg/.png)
and see the prediction, confidence and top 3 results.
Stop the app with Ctrl+C in the terminal.


## Project report ideas

- Explain transfer learning and why MobileNetV2 (lightweight, fast)
- Show the dataset class distribution
- Add accuracy/loss graphs and a confusion matrix
- Discuss limitations (lab-style images vs real field photos)
- Future work: treatment advice in Marathi/Hindi, mobile app, field-photo dataset
