import os
import torch
from PIL import Image
import numpy as np
import torchvision.transforms as T
from u2net import U2NET  # Assuming U2NET is defined in u2net.py

# Helper function to load the model
def load_model(model, model_path, device):
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model

# Preprocessing function to transform image
def preprocess_image(image_path, resize_shape=(320, 320)):
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])
    transforms = T.Compose([
        T.Resize(resize_shape),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std)
    ])
    image = Image.open(image_path).convert("RGB")
    image = transforms(image).unsqueeze(0)  # Add batch dimension
    return image

# Inference function
def predict(model, image_tensor, device):
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        output = model(image_tensor)
        return output[0]  # Returning only the first prediction (fused output)

# Function to remove background and create output image
# Function to remove the background and save the image with transparency
def remove_background(original_image_path, prediction_mask, save_as_png=True):
    # Load original image
    original_image = Image.open(original_image_path).convert("RGB")
    
    # Convert the prediction mask to NumPy array and then resize
    prediction_mask_np = prediction_mask.squeeze().cpu().numpy()  # Convert tensor to NumPy array
    prediction_mask_resized = Image.fromarray((prediction_mask_np * 255).astype('uint8')).resize(original_image.size, Image.BILINEAR)
    
    # Convert the resized mask to a binary mask (0 or 255)
    alpha_channel = np.array(prediction_mask_resized)
    alpha_channel = np.where(alpha_channel > 128, 255, 0).astype(np.uint8)

    # Combine original image with alpha channel
    original_image_np = np.array(original_image)
    rgba_image = np.dstack((original_image_np, alpha_channel))  # Add alpha channel

    # Save the resulting image
    output_path = original_image_path.replace('.jpg', '_no_background.png') if save_as_png else 'output_image.jpg'
    Image.fromarray(rgba_image).save(output_path)

    print(f"Image with background removed saved to: {output_path}")

# Main function
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the model
    u2net_model = U2NET(in_ch=3, out_ch=1)
    u2net_model = load_model(u2net_model, r'C:\Users\16292\Documents\AI_OUTFIT_PROJECT\models\u2net.pth', device)

    # Load and preprocess the image
    image_path = r'C:\Users\16292\Documents\AI_OUTFIT_PROJECT\Personal_Wardrobe_Images\user_black_red_shorts.jpg'  # Provide the correct image path
    image_tensor = preprocess_image(image_path)

    # Run inference
    prediction = predict(u2net_model, image_tensor, device)

    # Resize prediction and remove background
    remove_background(image_path, prediction, save_as_png=True)  # Save as PNG with transparency

if __name__ == '__main__':
    main()
