import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
from sklearn.utils.class_weight import compute_class_weight

# GPU Memory Growth Fix (Prevents RTX 2050 crashes)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# -----------------------
# SETTINGS
# -----------------------
IMG_SIZE = 128
BATCH_SIZE = 32 
EPOCHS_PHASE1 = 20
EPOCHS_PHASE2 = 20
train_dir = "data/mri_images"

# -----------------------
# DATA AUGMENTATION
# -----------------------
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

val_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

# -----------------------
# CLASS WEIGHTS
# -----------------------
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(train_generator.classes),
    y=train_generator.classes
)
class_weights = dict(enumerate(class_weights))

# -----------------------
# MODEL: EfficientNetV2-S
# -----------------------
base_model = tf.keras.applications.EfficientNetV2S(
    include_top=False,
    weights="imagenet",
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.BatchNormalization(),
    layers.Dense(256, activation="gelu"), 
    layers.Dropout(0.4),
    layers.Dense(train_generator.num_classes, activation="softmax")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
    metrics=["accuracy"]
)

# -----------------------
# CALLBACKS (Syntax Fixed Here)
# -----------------------
callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=6, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-7),
    tf.keras.callbacks.ModelCheckpoint("models/best_model_v2.keras", save_best_only=True)
] # Fixed: changed ) to ]

# -----------------------
# PHASE 1: WARMUP
# -----------------------
print("🚀 Starting Phase 1...")
model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS_PHASE1,
    class_weight=class_weights,
    callbacks=callbacks
)

# -----------------------
# PHASE 2: FULL FINE-TUNING
# -----------------------
print("🔥 Starting Phase 2...")
base_model.trainable = True

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
    metrics=["accuracy"]
)

model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS_PHASE2,
    class_weight=class_weights,
    callbacks=callbacks
)

# -----------------------
# SAVE FINAL MODEL
# -----------------------
os.makedirs("models", exist_ok=True)
model.save("models/final_mri_model_v2.keras")
print("✅ Done!")