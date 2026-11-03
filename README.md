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

## Step 1: Install requirements

1. Install **Python 3.10, 3.11 or 3.12** (64-bit) from python.org.
   During install, tick **"Add Python to PATH"**.
   (TensorFlow does NOT work on Python 3.13+ in many setups, so avoid it.)
2. Install **VS Code** and its **Python** extension (publisher: Microsoft).

## Step 2: Open the project in VS Code

1. Unzip `plant-disease-detector.zip`.
2. VS Code -> **File -> Open Folder** -> select the `plant-disease-detector` folder.
3. Open the terminal: **Terminal -> New Terminal** (Ctrl + `).

## Step 3: Create a virtual environment

**Windows (PowerShell):**
```
python -m venv venv
venv\Scripts\activate
```
If activation is blocked, run once:
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` and try again.

**Mac / Linux:**
```
python3 -m venv venv
source venv/bin/activate
```

You should see `(venv)` at the start of the terminal line.
If VS Code asks to use the new environment, click **Yes**. Otherwise press
Ctrl+Shift+P -> **Python: Select Interpreter** -> choose the one inside `venv`.

## Step 4: Install libraries

```
pip install --upgrade pip
pip install -r requirements.txt
```
This takes a few minutes (TensorFlow is large).

## Step 5: Download the dataset

1. Go to Kaggle and search **"PlantVillage dataset"** (about 54,000 images, 38 classes).
   A free Kaggle account is needed.
2. Download and unzip it.
3. Move the class folders into the `PlantVillage/` folder of this project
   (see `PlantVillage/PUT_DATASET_HERE.txt`).
   Some versions contain `color`, `grayscale`, `segmented` subfolders. Use the
   **color** folder's contents.

Tip: for a quick test, keep only 3-5 classes first so training finishes fast.

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

## Troubleshooting

| Problem | Fix |
|---|---|
| `python` not recognized | Reinstall Python with "Add to PATH", restart VS Code |
| `No matching distribution found for tensorflow` | Use Python 3.10-3.12, 64-bit |
| `Found 0 files belonging to 0 classes` | Dataset folder is empty or nested one level too deep |
| `FileNotFoundError: plant_model.keras` | Run `python train.py` first |
| Training too slow | Reduce `EPOCHS` or use fewer classes / images |
| `streamlit` not recognized | Activate the venv first (`venv\Scripts\activate`) |
| Wrong predictions on phone photos | PlantVillage has clean backgrounds; use a close, well-lit photo of one leaf on a plain background |

## Settings you can change (in train.py)

- `EPOCHS = 5`: more epochs can improve accuracy
- `BATCH = 32`: lower to 16 if you run out of memory
- `DATA_DIR = "PlantVillage"`: dataset folder

## Project report ideas

- Explain transfer learning and why MobileNetV2 (lightweight, fast)
- Show the dataset class distribution
- Add accuracy/loss graphs and a confusion matrix
- Discuss limitations (lab-style images vs real field photos)
- Future work: treatment advice in Marathi/Hindi, mobile app, field-photo dataset
