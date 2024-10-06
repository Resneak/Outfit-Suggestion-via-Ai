# fastai_resnet34_model_export.py

# Import necessary libraries
from fastai.vision.all import *
import pandas as pd
import os
import numpy as np

# Define the top_k_accuracy function at the global level
def top_k_accuracy(inp, targ, k=3):
    "Computes the Top-k accuracy for classification"
    inp = inp.topk(k=k, dim=-1)[1]
    targ = targ.unsqueeze(dim=-1)
    return (inp == targ).any(dim=-1).float().mean()

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

    # Adjust batch size according to your GPU memory
    bs = 128  # Use the same batch size as during training

    # Define data augmentations
    dblock = DataBlock(
        blocks=(ImageBlock, CategoryBlock),
        get_x=ColReader('full_path'),
        get_y=ColReader('label_name'),
        splitter=RandomSplitter(valid_pct=0.2, seed=42),
        item_tfms=Resize(460),
        batch_tfms=aug_transforms(size=224)
    )

    # Create DataLoaders
    dls = dblock.dataloaders(category_img, bs=bs, num_workers=0, pin_memory=True)

    # Define the learner using a pre-trained ResNet34 model
    learn = vision_learner(
        dls,
        resnet34,
        metrics=[accuracy,
                 partial(top_k_accuracy, k=3),
                 partial(top_k_accuracy, k=5)]
    )

    # Load the best model weights
    print("Loading the best model weights...")
    learn.load('best_model')

    # Evaluate the model (optional)
    print("Evaluating the model...")
    val_results = learn.validate()
    metrics_names = ['Loss', 'Top-1 Acc', 'Top-3 Acc', 'Top-5 Acc']
    print(f"Validation Results: {dict(zip(metrics_names, val_results))}")

    # Save the model
    print("Saving the trained model...")
    learn.export('deepfashion_resnet34.pkl')

if __name__ == '__main__':
    main()
