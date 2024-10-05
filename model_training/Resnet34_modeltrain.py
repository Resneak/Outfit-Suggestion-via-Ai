# fastai_resnet34_modeltrain.py

# Import necessary libraries
from fastai.vision.all import *
import pandas as pd
import os
import numpy as np
from functools import partial

def main():
    # Define the base path to the dataset
    DATASET_BASE_PATH = "C:/Users/16292/Documents/AI_OUTFIT_PROJECT/deepfashion_dataset/"

    # Ensure the dataset path exists
    assert os.path.exists(DATASET_BASE_PATH), f"Dataset path {DATASET_BASE_PATH} does not exist."

    # Load the image category labels
    category_img_path = os.path.join(DATASET_BASE_PATH, 'Anno_coarse', 'list_category_img.txt')
    category_img = pd.read_csv(category_img_path, sep='\s+', header=1)

    # Adjust category labels to start from 0
    category_img['category_label'] = category_img['category_label'] - 1

    # Load category names
    category_names_path = os.path.join(DATASET_BASE_PATH, 'Anno_coarse', 'list_category_cloth.txt')
    category_names = pd.read_csv(category_names_path, sep='\s+', header=1)

    # Map category labels to category names
    label_mapping = dict(zip(range(len(category_names)), category_names['category_name']))
    category_img['label_name'] = category_img['category_label'].map(label_mapping)

    # Create a full image path
    category_img['full_path'] = category_img['image_name'].apply(lambda x: os.path.join(DATASET_BASE_PATH, x))

    # Remove entries with missing images (if any)
    category_img = category_img[category_img['full_path'].apply(os.path.exists)].reset_index(drop=True)

    # Display the number of classes and samples
    num_classes = category_img['label_name'].nunique()
    print(f"Number of classes: {num_classes}")
    print(f"Number of samples: {len(category_img)}")

    # Split data into training and validation sets using RandomSplitter
    dblock = DataBlock(
        blocks=(ImageBlock, CategoryBlock),
        get_x=ColReader('full_path'),
        get_y=ColReader('label_name'),
        splitter=RandomSplitter(valid_pct=0.2, seed=42),
        item_tfms=Resize(460),
        batch_tfms=[
            *aug_transforms(size=224, min_scale=0.75),
            Normalize.from_stats(*imagenet_stats)
        ]
    )

    # Create DataLoaders with optimized batch size
    bs = 128  # Adjust batch size according to your GPU memory, you can try 512 if you want.
    dls = dblock.dataloaders(category_img, bs=bs, num_workers=0, pin_memory=True)  # Set num_workers=0 to avoid multiprocessing issues

    # Define the learner using a pre-trained ResNet34 model
    learn = vision_learner(dls, resnet34, metrics=accuracy)

    # Find the optimal learning rate
    print("Finding the optimal learning rate...")
    lr_min, lr_steep = learn.lr_find(suggest_funcs=(minimum, steep))

    print(f"Suggested learning rates - Minimum/10: {lr_min:.2e}, Steepest gradient: {lr_steep:.2e}")

    # Choose the learning rate
    lr = lr_steep

    # Implement Early Stopping and SaveModel callbacks
    from fastai.callback.tracker import SaveModelCallback, EarlyStoppingCallback

    # Define callbacks with increased patience for longer training
    callbacks = [
        SaveModelCallback(monitor='accuracy', fname='best_model'),
        EarlyStoppingCallback(monitor='accuracy', patience=5)  # Increased patience to allow more fluctuation before stopping
    ]

    # Train the model with fine-tuning using discriminative learning rates
    print("Starting training with fine-tuning...")
    learn.fine_tune(
        epochs=100,  # Set high number of epochs for overnight training
        base_lr=lr,
        freeze_epochs=3,  # Number of epochs to train with frozen layers
        cbs=callbacks
    )

    # Unfreeze the model for additional training with discriminative learning rates
    print("Unfreezing the model for further training...")
    learn.unfreeze()

    # Set discriminative learning rates (lower for early layers, higher for later layers)
    lr_max = lr
    lr_min = lr / 10
    print(f"Using discriminative learning rates: lr_min={lr_min:.2e}, lr_max={lr_max:.2e}")

    # Continue training with more epochs
    learn.fit_one_cycle(
        100,  # Higher number of epochs with early stopping
        lr_max=slice(lr_min, lr_max),
        cbs=callbacks
    )

    # Save the model after training
    print("Saving the trained model...")
    learn.export('deepfashion_resnet34.pkl')

if __name__ == '__main__':
    main()
