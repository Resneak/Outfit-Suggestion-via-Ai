from fastai.vision.all import *

import os



correct_dir = r"C:\Users\16292\Documents\AI_OUTFIT_PROJECT\model_training"
model_path = os.path.join(correct_dir, 'deepfashion_resnet34.pkl')



# Load the exported model
learn = load_learner(model_path) # 

# Test on a sample image
img = PILImage.create(r'C:\Users\16292\Documents\AI_OUTFIT_PROJECT\Personal_Wardrobe_Images\red_shoes.png')
pred_class, pred_idx, outputs = learn.predict(img)
print(f"Predicted class: {pred_class}")
