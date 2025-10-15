import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications import (
    EfficientNetB0,
    ResNet50,
    InceptionV3
)

# ---------------------------------------------------
# 1. CONFIG
# ---------------------------------------------------
data_dir = "dataset"
img_size = (224, 224)
batch_size = 32
num_classes = 2
epochs = 10
model_name = "resnet"  # "efficientnet" or "inception"
learning_rate = 1e-4

# ---------------------------------------------------
# 2. DATA LOADING
# ---------------------------------------------------
train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(
    rescale=1./255
)

train_gen = train_datagen.flow_from_directory(
    data_dir + "/train",
    target_size=img_size,
    batch_size=batch_size,
    class_mode="categorical"
)

val_gen = val_datagen.flow_from_directory(
    data_dir + "/val",
    target_size=img_size,
    batch_size=batch_size,
    class_mode="categorical"
)

# ---------------------------------------------------
# 3. MODEL SELECTOR
# ---------------------------------------------------
def get_model(name):
    if name == "efficientnet":
        base = EfficientNetB0(weights="imagenet", include_top=False, input_shape=img_size + (3,))
    elif name == "resnet":
        base = ResNet50(weights="imagenet", include_top=False, input_shape=img_size + (3,))
    elif name == "inception":
        base = InceptionV3(weights="imagenet", include_top=False, input_shape=img_size + (3,))
    else:
        raise ValueError("Choose from: resnet, efficientnet, inception")

    base.trainable = False  # Freeze backbone

    model = models.Sequential([
        base,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation="softmax")
    ])
    return model

model = get_model(model_name)

model.compile(
    optimizer=optimizers.Adam(learning_rate),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# ---------------------------------------------------
# 4. TRAINING
# ---------------------------------------------------
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=epochs
)

# ---------------------------------------------------
# 5. SAVE MODEL
# ---------------------------------------------------
model.save(f"best_{model_name}.h5")
