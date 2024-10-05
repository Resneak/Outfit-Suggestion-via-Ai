import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import mixed_precision
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

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

# Create training, validation, and test sets
train_data = data[data['evaluation_status'] == 'train'].reset_index(drop=True)
val_data = data[data['evaluation_status'] == 'val'].reset_index(drop=True)
test_data = data[data['evaluation_status'] == 'test'].reset_index(drop=True)

# Define the image preprocessing function with augmentation
def load_and_preprocess_image_with_augmentation(row):
    image_path = tf.strings.join([DATASET_BASE_PATH, row['image_name']], separator=os.sep)
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    
    x1 = tf.cast(row['x_1'], tf.int32)
    y1 = tf.cast(row['y_1'], tf.int32)
    x2 = tf.cast(row['x_2'], tf.int32)
    y2 = tf.cast(row['y_2'], tf.int32)
    
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
    
    return img_aug

# Original image preprocessing function
def load_and_preprocess_image(row):
    image_path = tf.strings.join([DATASET_BASE_PATH, row['image_name']], separator=os.sep)
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    
    x1 = tf.cast(row['x_1'], tf.int32)
    y1 = tf.cast(row['y_1'], tf.int32)
    x2 = tf.cast(row['x_2'], tf.int32)
    y2 = tf.cast(row['y_2'], tf.int32)
    
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
    
    return img_normalized

# Create TensorFlow datasets
def create_tf_dataset(data, batch_size, shuffle=True, augment=False):
    image_names = tf.constant(data['image_name'].values)
    category_labels = tf.constant(data['category_label'].values, dtype=tf.int32)
    x1 = tf.constant(data['x_1'].values, dtype=tf.int32)
    y1 = tf.constant(data['y_1'].values, dtype=tf.int32)
    x2 = tf.constant(data['x_2'].values, dtype=tf.int32)
    y2 = tf.constant(data['y_2'].values, dtype=tf.int32)
    
    dataset = tf.data.Dataset.from_tensor_slices({
        'image_name': image_names,
        'category_label': category_labels,
        'x_1': x1,
        'y_1': y1,
        'x_2': x2,
        'y_2': y2
    })
    
    if shuffle:
        dataset = dataset.shuffle(buffer_size=10000)
    
    if augment:
        dataset = dataset.map(lambda row: (load_and_preprocess_image_with_augmentation(row), row['category_label']), num_parallel_calls=tf.data.AUTOTUNE)
    else:
        dataset = dataset.map(lambda row: (load_and_preprocess_image(row), row['category_label']), num_parallel_calls=tf.data.AUTOTUNE)
    
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
    # Build the model
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    
    # Fine-tune the base model
    base_model.trainable = True
    for layer in base_model.layers[:-50]:
        layer.trainable = False
    
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(category_cloth.shape[0], activation='softmax', dtype='float32')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # Compile the model with a lower learning rate
    optimizer = Adam(learning_rate=1e-5)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    # Define callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2)
    model_checkpoint = ModelCheckpoint(filepath=model_path, save_best_only=True)
    callbacks = [early_stopping, reduce_lr, model_checkpoint]
    
    # Train the model
    epochs = 10  # Increase the number of epochs
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        callbacks=callbacks
    )
    
    print("Model trained and saved to disk.")

# Evaluate the model
val_loss, val_accuracy = model.evaluate(val_dataset)
print(f"\nValidation Loss: {val_loss}")
print(f"Validation Accuracy: {val_accuracy}")

# Plot training & validation accuracy and loss only if 'history' exists
if 'history' in locals():
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
