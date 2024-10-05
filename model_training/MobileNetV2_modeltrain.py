import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import mixed_precision
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Enable mixed precision
mixed_precision.set_global_policy('mixed_float16')

# Check GPU availability
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# Define the base path to the dataset
DATASET_BASE_PATH = "C:/Users/16292/Documents/AI_OUTFIT_PROJECT/deepfashion_dataset/"

# Load category names and types
category_cloth = pd.read_csv(
    os.path.join(DATASET_BASE_PATH, 'Anno_coarse/list_category_cloth.txt'),
    sep='\s+', header=1
)

# Load image category labels
category_img = pd.read_csv(
    os.path.join(DATASET_BASE_PATH, 'Anno_coarse/list_category_img.txt'),
    sep='\s+', header=1
)

# Load bounding boxes
bbox = pd.read_csv(
    os.path.join(DATASET_BASE_PATH, 'Anno_coarse/list_bbox.txt'),
    sep='\s+', header=1
)

# Load evaluation partitions
eval_partition = pd.read_csv(
    os.path.join(DATASET_BASE_PATH, 'Eval/list_eval_partition.txt'),
    sep='\s+', header=1
)

# Merge dataframes
data = pd.merge(category_img, eval_partition, on='image_name')
data = pd.merge(data, bbox, on='image_name')

# Adjust category labels to start from 0
data['category_label'] = data['category_label'] - 1

# Ensure that labels are mapped to a continuous range
unique_labels = sorted(data['category_label'].unique())
label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
data['category_label_mapped'] = data['category_label'].map(label_mapping)

# Print number of unique labels and the label mapping
num_classes = len(unique_labels)
print(f"Number of classes: {num_classes}")
print("Unique labels in the dataset:", unique_labels)

# Create training, validation, and test sets
train_data = data[data['evaluation_status'] == 'train'].reset_index(drop=True)
val_data = data[data['evaluation_status'] == 'val'].reset_index(drop=True)
test_data = data[data['evaluation_status'] == 'test'].reset_index(drop=True)

# Define the image preprocessing function with augmentation
def load_and_preprocess_image_with_augmentation(image_name, category_label, x1, y1, x2, y2):
    image_path = tf.strings.join([DATASET_BASE_PATH, image_name], separator=os.sep)
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    
    x1 = tf.cast(x1, tf.int32)
    y1 = tf.cast(y1, tf.int32)
    x2 = tf.cast(x2, tf.int32)
    y2 = tf.cast(y2, tf.int32)
    
    shape = tf.shape(img)
    h = shape[0]
    w = shape[1]
    x1 = tf.clip_by_value(x1, 0, w)
    y1 = tf.clip_by_value(y1, 0, h)
    x2 = tf.clip_by_value(x2, 0, w)
    y2 = tf.clip_by_value(y2, 0, h)
    
    img_cropped = img[y1:y2, x1:x2]
    img_resized = tf.image.resize(img_cropped, [224, 224])
    img_normalized = img_resized / 255.0

    # Apply data augmentation
    img_aug = tf.image.random_flip_left_right(img_normalized)
    img_aug = tf.image.random_brightness(img_aug, max_delta=0.1)
    img_aug = tf.image.random_contrast(img_aug, lower=0.9, upper=1.1)
    img_aug = tf.image.random_saturation(img_aug, lower=0.9, upper=1.1)
    
    return img_aug, category_label

# Original image preprocessing function
def load_and_preprocess_image(image_name, category_label, x1, y1, x2, y2):
    image_path = tf.strings.join([DATASET_BASE_PATH, image_name], separator=os.sep)
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    
    x1 = tf.cast(x1, tf.int32)
    y1 = tf.cast(y1, tf.int32)
    x2 = tf.cast(x2, tf.int32)
    y2 = tf.cast(y2, tf.int32)
    
    shape = tf.shape(img)
    h = shape[0]
    w = shape[1]
    x1 = tf.clip_by_value(x1, 0, w)
    y1 = tf.clip_by_value(y1, 0, h)
    x2 = tf.clip_by_value(x2, 0, w)
    y2 = tf.clip_by_value(y2, 0, h)
    
    img_cropped = img[y1:y2, x1:x2]
    img_resized = tf.image.resize(img_cropped, [224, 224])
    img_normalized = img_resized / 255.0
    
    return img_normalized, category_label

# Create TensorFlow datasets
def create_tf_dataset(data, batch_size, shuffle=True, augment=False):
    dataset = tf.data.Dataset.from_tensor_slices({
        'image_name': tf.constant(data['image_name'].values),
        'category_label': tf.constant(data['category_label_mapped'].values.astype(np.int32)),
        'x_1': tf.constant(data['x_1'].values, dtype=tf.int32),
        'y_1': tf.constant(data['y_1'].values, dtype=tf.int32),
        'x_2': tf.constant(data['x_2'].values, dtype=tf.int32),
        'y_2': tf.constant(data['y_2'].values, dtype=tf.int32)
    })
    
    def process_row(row):
        if augment:
            return load_and_preprocess_image_with_augmentation(row['image_name'], row['category_label'], row['x_1'], row['y_1'], row['x_2'], row['y_2'])
        else:
            return load_and_preprocess_image(row['image_name'], row['category_label'], row['x_1'], row['y_1'], row['x_2'], row['y_2'])
    
    dataset = dataset.map(process_row, num_parallel_calls=tf.data.AUTOTUNE)
    
    if shuffle:
        dataset = dataset.shuffle(buffer_size=10000)
    
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

# Set the batch size
batch_size = 128  # Adjust based on your GPU memory

# Create datasets
train_dataset = create_tf_dataset(train_data, batch_size=batch_size, shuffle=True, augment=True)
val_dataset = create_tf_dataset(val_data, batch_size=batch_size, shuffle=False, augment=False)

# Update the path to save the model within the current project folder
model_save_dir = "C:/Users/16292/Documents/AI_OUTFIT_PROJECT/models"
if not os.path.exists(model_save_dir):
    os.makedirs(model_save_dir)

# Check if model already exists
model_path = os.path.join(model_save_dir, 'clothing_classifier.h5')

if os.path.exists(model_path):
    # Load the saved model
    model = load_model(model_path)
    print("Model loaded from disk.")
else:
    # Build the model using ResNet50
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    # Freeze all layers in the base model
    for layer in base_model.layers:
        layer.trainable = False

    # Add new classification layers
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    # Create the full model
    model = Model(inputs=base_model.input, outputs=predictions)

    # Compile the model
    optimizer = Adam(learning_rate=1e-4)  # Lower learning rate to start
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Train only the top layers (which were randomly initialized)
    initial_epochs = 10
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=initial_epochs
    )

    # Unfreeze all layers for fine-tuning
    for layer in model.layers:
        layer.trainable = True

    # Recompile the model with a lower learning rate
    optimizer = Adam(learning_rate=1e-5)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Define callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint(filepath=model_path, save_best_only=True)
    callbacks = [early_stopping, model_checkpoint]

    # Continue training the entire model
    fine_tune_epochs = 30
    total_epochs = initial_epochs + fine_tune_epochs

    history_fine = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=total_epochs,
        initial_epoch=history.epoch[-1],
        callbacks=callbacks
    )

    print("Model trained and saved to disk.")

# Evaluate the model
val_loss, val_accuracy = model.evaluate(val_dataset)
print(f"\nValidation Loss: {val_loss}")
print(f"Validation Accuracy: {val_accuracy}")

# Plot training & validation accuracy and loss
if 'history_fine' in locals():
    history.history['accuracy'] += history_fine.history['accuracy']
    history.history['val_accuracy'] += history_fine.history['val_accuracy']
    history.history['loss'] += history_fine.history['loss']
    history.history['val_loss'] += history_fine.history['val_loss']

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='upper left')

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper left')

    plt.tight_layout()
    plt.show()
