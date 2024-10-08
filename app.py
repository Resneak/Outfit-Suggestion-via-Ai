# app.py

from flask import Flask, render_template, request, redirect, url_for, jsonify
import requests
import os
import cv2
import numpy as np
from sklearn.cluster import KMeans
from collections import Counter
from flask_sqlalchemy import SQLAlchemy
import webcolors
from colormath.color_objects import sRGBColor, LabColor
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie2000

import colorsys
from colormath.color_objects import sRGBColor, LabColor
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie2000


from fastai.vision.all import load_learner, PILImage
from functools import partial
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

from u2net import U2NET  # Assuming the u2net.py file is in the same directory
import torchvision.transforms as transforms


app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Configure the SQLite database
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False  # Silence the deprecation warning
db = SQLAlchemy(app)  # Initialize the database

# Ensure the upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

OPENWEATHER_API_KEY = os.environ.get('OPENWEATHER_API_KEY')
if not OPENWEATHER_API_KEY:
    raise RuntimeError("OpenWeather API key not found. Please set the OPENWEATHER_API_KEY environment variable.")

# Define the top_k_accuracy function to resolve the error
def top_k_accuracy(inp, targ, k=3):
    "Computes the Top-k accuracy for classification"
    inp = inp.topk(k=k, dim=-1)[1]
    targ = targ.unsqueeze(dim=-1)
    return (inp == targ).any(dim=-1).float().mean()

# Load the trained clothing classification model
MODEL_PATH = './models/deepfashion_resnet34.pkl'
#learn = load_learner(MODEL_PATH)

# for heroku
learn = load_learner(MODEL_PATH, cpu=True)



# Get the list of detailed categories from the model's data
if learn and hasattr(learn.dls, 'vocab'):
    DETAILED_CATEGORIES = learn.dls.vocab
else:
    DETAILED_CATEGORIES = []  # Fallback if model loading fails

# Load U-2-Net model for background removal
U2NET_PATH = 'u2net.pth'  # Ensure the u2net.pth file is in the same directory
u2net = U2NET(3, 1)
u2net.load_state_dict(torch.load(U2NET_PATH, map_location=torch.device('cpu')))
u2net.eval()  # Set model to evaluation mode

# Define the mapping from detailed categories to broader categories
CATEGORY_MAPPING = {
    # Top categories
    'Anorak': 'Top',
    'Blazer': 'Top',
    'Blouse': 'Top',
    'Bomber': 'Top',
    'Button-Down': 'Top',
    'Cardigan': 'Top',
    'Flannel': 'Top',
    'Halter': 'Top',
    'Henley': 'Top',
    
    'Jersey': 'Top',
    'Sweater': 'Top',
    'Tank': 'Top',
    'Tee': 'Top',
    'Top': 'Top',
    'Turtleneck': 'Top',

    # Bottom categories
    'Capris': 'Bottom',
    'Chinos': 'Bottom',
    'Culottes': 'Bottom',
    'Cutoffs': 'Bottom',
    'Gauchos': 'Bottom',
    'Jeans': 'Bottom',
    'Jeggings': 'Bottom',
    'Jodhpurs': 'Bottom',
    'Joggers': 'Bottom',
    'Leggings': 'Bottom',
    'Sarong': 'Bottom',
    'Shorts': 'Bottom',
    'Skirt': 'Bottom',
    'Sweatpants': 'Bottom',
    'Sweatshorts': 'Bottom',
    'Trunks': 'Bottom',

    # Outerwear categories
    'Hoodie': 'Outerwear',
    'Parka': 'Outerwear',
    'Peacoat': 'Outerwear',
    'Poncho': 'Outerwear',
    'Caftan': 'Outerwear',
    'Cape': 'Outerwear',
    'Coat': 'Outerwear',
    'Jacket': 'Outerwear',

    # Dress/Full Body categories
    'Coverup': 'Dress',
    'Dress': 'Dress',
    'Jumpsuit': 'Dress',
    'Kaftan': 'Dress',
    'Kimono': 'Dress',
    'Nightdress': 'Dress',
    'Onesie': 'Dress',
    'Robe': 'Dress',
    'Romper': 'Dress',
    'Shirtdress': 'Dress',
    'Sundress': 'Dress'
}

# Define the list of broader categories
CATEGORY_LIST = ['Top', 'Bottom', 'Outerwear', 'Footwear', 'Accessory', 'Dress']

class ClothingItem(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    image_filename = db.Column(db.String(100), nullable=False)
    colors = db.Column(db.String(500), nullable=False) 
    category = db.Column(db.String(50), nullable=False)
    detailed_category = db.Column(db.String(50), nullable=False)

    def __repr__(self):
        return f'<ClothingItem {self.id}>'

with app.app_context():
    db.create_all()

# -+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+ ROUTES

@app.route('/')
def home():
    items = ClothingItem.query.all()
    return render_template('index.html', items=items)

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'Missing file', 400

        file = request.files['file']

        if file.filename == '':
            return 'No selected file', 400

        if file:
            filename = file.filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Remove background and save the processed image
            processed_image = remove_background(filepath)
            processed_filename = f"processed_{os.path.splitext(filename)[0]}.png"
            processed_filepath = os.path.join(app.config['UPLOAD_FOLDER'], processed_filename)
            processed_image.save(processed_filepath, format='PNG')

            # Process the image for color extraction
            colors = extract_colors(processed_filepath)

            # Predict category using the model
            if learn:
                try:
                    pred_class, pred_idx, probs = learn.predict(filepath)
                    confidence = round(probs[pred_idx].item() * 100)
                    detailed_category = pred_class
                    broader_category = CATEGORY_MAPPING.get(detailed_category, 'Accessory')
                    print(f"Predicted Category: {detailed_category} -> {broader_category}, Confidence: {confidence:.2f}%")
                except Exception as e:
                    print(f"Error during prediction: {e}")
                    detailed_category = "Unknown"
                    broader_category = "Accessory"
                    confidence = 0
            else:
                detailed_category = "Unknown"
                broader_category = "Accessory"
                confidence = 0

            # Render a template to display the images, extracted colors, and predicted category
            return render_template(
                'confirm.html',
                colors=colors,
                filename=filename,
                processed_filename=processed_filename,
                predicted_category=detailed_category,
                broader_category=broader_category,
                categories=CATEGORY_LIST,
                confidence=confidence
            )
    else:
        # If GET request, render the upload page
        return render_template('upload.html')

@app.route('/delete_item/<int:item_id>', methods=['POST'])
def delete_item(item_id):
    item = ClothingItem.query.get_or_404(item_id)
    db.session.delete(item)
    db.session.commit()
    return redirect(url_for('wardrobe'))

@app.route('/confirm', methods=['POST'])
def confirm():
    # Get data from the form submission
    filename = request.form.get('filename')
    category = request.form.get('category')
    detailed_category = request.form.get('detailed_category', 'Unknown')

    if not filename or not category:
        return 'Missing data', 400

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    # Extract colors
    colors = extract_colors(filepath)

    # Save to database
    colors_str = ','.join(colors)
    new_item = ClothingItem(
        image_filename=filename,
        colors=colors_str,
        category=category,
        detailed_category=detailed_category
    )
    db.session.add(new_item)
    db.session.commit()

    # Redirect to wardrobe or render a success page
    return redirect(url_for('wardrobe'))

@app.route('/filter_by_color/<color>')
def filter_by_color(color):
    items = ClothingItem.query.filter(ClothingItem.colors.like(f'%{color}%')).all()
    all_colors = set()
    for item in items:
        all_colors.update(item.colors.split(','))
    return render_template('wardrobe.html', items=items, all_colors=sorted(all_colors))

@app.route('/wardrobe')
def wardrobe():
    items = ClothingItem.query.all()
    all_colors = set()
    for item in items:
        all_colors.update(item.colors.split(','))
    return render_template('wardrobe.html', items=items, all_colors=sorted(all_colors))

@app.route('/create_outfit', methods=['GET', 'POST'])
def create_outfit():
    if request.method == 'POST':
        selected_item_ids = []
        for category in CATEGORY_LIST:
            item_id = request.form.get('selected_items_' + category)
            if item_id:
                selected_item_ids.append(int(item_id))
        if selected_item_ids:
            selected_items = ClothingItem.query.filter(ClothingItem.id.in_(selected_item_ids)).all()
        else:
            selected_items = []
        # Organize the selected items by category
        selected_items_by_category = {category: None for category in CATEGORY_LIST}
        for item in selected_items:
            selected_items_by_category[item.category] = item
        return render_template('display_outfit.html', selected_items=selected_items_by_category, CATEGORY_LIST=CATEGORY_LIST)
    else:
        # GET request, display the selection form
        items = ClothingItem.query.all()
        categories = CATEGORY_LIST
        items_by_category = {category: [] for category in categories}
        for item in items:
            items_by_category[item.category].append(item)
        return render_template('create_outfit.html', items_by_category=items_by_category)


@app.route('/suggestions', methods=['GET'])
def suggestions():
    # Get selected categories from the query parameters
    selected_categories = request.args.getlist('categories')

    # If no categories are selected, default to mandatory categories
    if not selected_categories:
        selected_categories = ['Top', 'Bottom', 'Footwear']  # Default categories

    # Call the suggestion function with selected categories
    suggestions = suggest_outfits(selected_categories)

    return render_template('suggestions.html', suggestions=suggestions, selected_categories=selected_categories)

@app.route('/get_weather', methods=['POST'])
def get_weather():
    data = request.get_json()
    lat = data.get('lat')
    lon = data.get('lon')

    if not lat or not lon:
        return jsonify({'error': 'Missing latitude or longitude'}), 400

    # Build the API URL
    url = 'https://api.openweathermap.org/data/2.5/weather'
    params = {
        'lat': lat,
        'lon': lon,
        'appid': OPENWEATHER_API_KEY,
        'units': 'imperial'  # Use 'imperial' for Fahrenheit
    }

    # Make the API request
    response = requests.get(url, params=params)

    if response.status_code == 200:
        weather_data = response.json()
        # Extract weather information
        description = weather_data['weather'][0]['description'].capitalize()
        temp = round(weather_data['main']['temp'])
        icon_code = weather_data['weather'][0]['icon']
        # Build the icon URL
        icon_url = f"http://openweathermap.org/img/wn/{icon_code}@2x.png"
        # Prepare data to send to the client
        weather_info = {
            'description': description,
            'temp': temp,
            'icon_url': icon_url
        }
        return jsonify(weather_info)
    else:
        print(f"OpenWeather API Error: {response.status_code} - {response.text}")
        return jsonify({'error': 'Failed to retrieve weather data'}), response.status_code

# -+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+ FUNCTIONS

def predict_category(image_path):
    img = PILImage.create(image_path)
    pred_class, pred_idx, outputs = learn.predict(img)
    detailed_category = str(pred_class)
    # Map to broader category
    broader_category = CATEGORY_MAPPING.get(detailed_category, 'Accessory')  # Default to 'Accessory' if not found
    return detailed_category, broader_category

def get_lab_color(hex_color):
    rgb = sRGBColor.new_from_rgb_hex(hex_color)
    lab = convert_color(rgb, LabColor)
    return lab

def calculate_color_difference(color1, color2):
    lab1 = get_lab_color(color1)
    lab2 = get_lab_color(color2)
    delta_e = delta_e_cie2000(lab1, lab2)
    return delta_e

def suggest_outfits(selected_categories):
    # Get all items from the database
    items = ClothingItem.query.all()

    # Separate items by category
    categories = {cat: [] for cat in CATEGORY_LIST}
    for item in items:
        if item.category in categories:
            categories[item.category].append(item)

    suggestions = []

    # Ensure that for selected categories, there are items available
    for cat in selected_categories:
        if not categories.get(cat):
            # No items in this category; cannot generate outfits
            return suggestions

    from itertools import product

    # Prepare a list of item lists for selected categories
    selected_items = [categories[cat] for cat in selected_categories]

    # Generate all combinations of selected items
    for combination in product(*selected_items):
        outfit = {}
        hex_colors = []
        for item in combination:
            category = item.category.lower()
            outfit[category] = item
            # Get the main color
            item_colors = [pair.split(',')[0] for pair in item.colors.split(';') if pair]
            if item_colors:
                hex_colors.append(item_colors[0])

        # Calculate average color difference between all pairs
        from itertools import combinations
        pair_diffs = []
        for color1, color2 in combinations(hex_colors, 2):
            diff = calculate_color_difference(color1, color2)
            pair_diffs.append(diff)

        if pair_diffs:
            avg_color_diff = sum(pair_diffs) / len(pair_diffs)
        else:
            avg_color_diff = 0

        # Decide whether to include the outfit based on color difference
        # Adjust thresholds as needed
        if 20 < avg_color_diff < 40:
            outfit['score'] = avg_color_diff
            suggestions.append(outfit)

    # Sort suggestions by score
    suggestions.sort(key=lambda x: x['score'])

    return suggestions

u2net_transform = transforms.Compose([
    transforms.Resize((320, 320)),  # Resize to the size expected by U-2-Net
    transforms.ToTensor(),  # Convert to Tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize using ImageNet stats
])

def remove_background(image_path):
    # Load the image
    image = Image.open(image_path).convert('RGB')
    original_size = image.size

    # Preprocess the image
    image_transformed = u2net_transform(image).unsqueeze(0)  # Add batch dimension

    # Run the image through the U-2-Net model
    with torch.no_grad():
        prediction = u2net(image_transformed)[0][:, 0, :, :].cpu().numpy()
    
    # Post-process the predicted mask
    mask = (prediction > 0.5).astype(np.uint8)  # Binarize the mask, threshold at 0.5

    # Resize mask to match the input image size
    mask = cv2.resize(mask[0], original_size, interpolation=cv2.INTER_NEAREST)

    # Convert image to numpy array
    image_np = np.array(image)

    # Apply the mask to the image, removing background
    result_image = image_np * mask[:, :, np.newaxis]

    # Create a transparent background
    result_image = cv2.cvtColor(result_image, cv2.COLOR_RGB2RGBA)
    result_image[:, :, 3] = mask * 255

    return Image.fromarray(result_image)




def closest_colour(requested_colour):
    min_colours = {}
    css3_names = webcolors.CSS3_NAMES_TO_HEX

    for name, hex_code in css3_names.items():
        r_c, g_c, b_c = webcolors.hex_to_rgb(hex_code)
        rd = (r_c - requested_colour[0]) ** 2
        gd = (g_c - requested_colour[1]) ** 2
        bd = (b_c - requested_colour[2]) ** 2
        min_colours[rd + gd + bd] = name

    closest_name = min_colours[min(min_colours.keys())]
    return closest_name

def get_color_name(hex_color):
    try:
        color_name = webcolors.hex_to_name(hex_color, spec='css3')
    except ValueError:
        # If the exact color name is not found, find the closest match
        rgb_color = webcolors.hex_to_rgb(hex_color)
        color_name = closest_colour(rgb_color)
    return color_name

def get_broad_color_name(rgb_color):
    r, g, b = [x/255.0 for x in rgb_color]
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    
    # Convert hue to degrees
    h *= 360
    
    # Adjust thresholds for better detection of purple and other colors
    if s < 0.15:  # Very low saturation colors
        if v > 0.9:
            return 'White'
        elif v < 0.2:
            return 'Black'
        else:
            return 'Gray'
    elif v < 0.2:
        return 'Black'
    elif h < 10 or h > 350:
        return 'Red'
    elif 10 <= h < 45:
        return 'Orange'
    elif 45 <= h < 70:
        return 'Yellow'
    elif 70 <= h < 165:
        return 'Green'
    elif 165 <= h < 190:
        return 'Cyan'
    elif 190 <= h < 270:  # Narrowed blue range
        return 'Blue'
    elif 270 <= h < 330:  # Expanded purple range
        return 'Purple'
    elif 330 <= h < 350:
        return 'Pink'
    else:
        return 'Unknown'

def extract_colors(image_path, num_colors=3):
    # Load the image
    img = Image.open(image_path)
    
    # Convert image to RGB numpy array
    img_np = np.array(img.convert('RGB'))

    # Reshape image to a list of pixels
    pixels = img_np.reshape((-1, 3))

    # If the original image was RGBA, create a mask for non-transparent pixels
    if img.mode == 'RGBA':
        alpha = np.array(img.split()[-1]).reshape((-1,))
        pixels = pixels[alpha != 0]
    
    # Check if there are enough pixels left after background removal
    if len(pixels) == 0:
        return ['White']

    # Use KMeans to cluster pixels
    kmeans = KMeans(n_clusters=num_colors)
    kmeans.fit(pixels)

    # Get colors and counts
    counts = Counter(kmeans.labels_)
    center_colors = kmeans.cluster_centers_

    # Sort colors by frequency
    ordered_colors = [center_colors[i] for i in counts.keys()]

    # Convert RGB to broad color names
    color_names = [get_broad_color_name(tuple(map(int, color))) for color in ordered_colors]

    # Remove duplicates while preserving order
    unique_color_names = []
    for color in color_names:
        if color not in unique_color_names and color != 'Unknown':
            unique_color_names.append(color)

    return unique_color_names[:num_colors]  # Return up to num_colors unique colors


def rgb_to_hex(color):
    return "#{:02x}{:02x}{:02x}".format(
        max(0, min(255, int(color[0]))),
        max(0, min(255, int(color[1]))),
        max(0, min(255, int(color[2])))
    )

if __name__ == '__main__':
    #app.run(debug=True)

    #for heroku
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
