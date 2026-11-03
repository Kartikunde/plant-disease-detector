"""
Plant Disease Detection - Training script
Dataset: PlantVillage (Kaggle) -> one folder per class, e.g.
    PlantVillage/
        Tomato___Late_blight/
        Tomato___healthy/
        Potato___Early_blight/
        ...
Run:  python train.py
"""
import json
import tensorflow as tf
from tensorflow.keras import layers

DATA_DIR = "PlantVillage"   # change to your dataset folder
IMG_SIZE = (224, 224)
BATCH = 32
EPOCHS = 5

train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR, validation_split=0.2, subset="training",
    seed=42, image_size=IMG_SIZE, batch_size=BATCH)
val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR, validation_split=0.2, subset="validation",
    seed=42, image_size=IMG_SIZE, batch_size=BATCH)

class_names = train_ds.class_names
print("Classes:", len(class_names))
with open("class_names.json", "w") as f:
    json.dump(class_names, f)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)

# Data augmentation helps the model handle real-world photos
augment = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])

# Transfer learning: pretrained MobileNetV2 as feature extractor
base = tf.keras.applications.MobileNetV2(
    input_shape=IMG_SIZE + (3,), include_top=False, weights="imagenet")
base.trainable = False

inputs = tf.keras.Input(shape=IMG_SIZE + (3,))
x = augment(inputs)
x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
x = base(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(len(class_names), activation="softmax")(x)
model = tf.keras.Model(inputs, outputs)

model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)

loss, acc = model.evaluate(val_ds)
print(f"Validation accuracy: {acc:.2%}")

model.save("plant_model.keras")
print("Saved plant_model.keras and class_names.json")
