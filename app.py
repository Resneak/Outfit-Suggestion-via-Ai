# app.py

from flask import Flask, render_template, request, redirect, url_for, jsonify
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
import requests

from fastai.vision.all import load_learner, PILImage
from functools import partial
import torch

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

# Load the trained model when the app starts
MODEL_PATH = './models/deepfashion_resnet34.pkl'
learn = load_learner(MODEL_PATH)

# Get the list of detailed categories from the model's data
if learn and hasattr(learn.dls, 'vocab'):
    DETAILED_CATEGORIES = learn.dls.vocab
else:
    DETAILED_CATEGORIES = []  # Fallback if model loading fails

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
    'Hoodie': 'Top',
    'Jacket': 'Top',
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
    'Parka': 'Outerwear',
    'Peacoat': 'Outerwear',
    'Poncho': 'Outerwear',
    'Caftan': 'Outerwear',
    'Cape': 'Outerwear',
    'Coat': 'Outerwear',

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
    category = db.Column(db.String(50), nullable=False)  # Broader category
    detailed_category = db.Column(db.String(50), nullable=False)  # Detailed category

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

            # Process the image for color extraction
            colors = extract_colors(filepath)

            # Predict category using the model
            if learn:
                try:
                    pred_class, pred_idx, probs = learn.predict(filepath)
                    confidence = round(probs[pred_idx].item() * 100)  # Convert to percentage and round to nearest whole number
                    detailed_category = pred_class
                    broader_category = CATEGORY_MAPPING.get(detailed_category, 'Accessory')
                    print(f"Predicted Category: {detailed_category} -> {broader_category}, Confidence: {confidence:.2f}%")
                except Exception as e:
                    print(f"Error during prediction: {e}")
                    detailed_category = "Unknown"
                    broader_category = "Accessory"  # Default to 'Accessory' if prediction fails
                    confidence = 0
            else:
                detailed_category = "Unknown"
                broader_category = "Accessory"
                confidence = 0

            # Render a template to display the image, extracted colors, and predicted category
            return render_template(
                'confirm.html',
                colors=colors,
                filename=filename,
                predicted_category=detailed_category,
                broader_category=broader_category,
                categories=CATEGORY_LIST,
                confidence=confidence
            )
    else:
        # If GET request, render the upload page
        return render_template('upload.html')


@app.route('/confirm', methods=['POST'])
def confirm():
    # Get data from the form submission
    filename = request.form.get('filename')
    category = request.form.get('category')
    detailed_category = request.form.get('detailed_category', 'Unknown')

    if not filename or not category:
        return 'Missing data', 400

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    # Re-extract colors in case this route is accessed directly
    colors = extract_colors(filepath)

    # Save to database
    colors_str = ';'.join([f"{hex_color},{color_name}" for hex_color, color_name in colors])
    new_item = ClothingItem(
        image_filename=filename,
        colors=colors_str,
        category=category,  # Broader category selected by the user
        detailed_category=detailed_category
    )
    db.session.add(new_item)
    db.session.commit()

    # Redirect to wardrobe or render a success page
    return redirect(url_for('wardrobe'))

# Route to delete a clothing item
@app.route('/delete_item/<int:item_id>', methods=['POST'])
def delete_item(item_id):
    item = ClothingItem.query.get(item_id)
    if item:
        db.session.delete(item)
        db.session.commit()
    return redirect(url_for('wardrobe'))

@app.route('/wardrobe')
def wardrobe():
    items = ClothingItem.query.all()
    return render_template('wardrobe.html', items=items)

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

def remove_background(image_path):
    img = cv2.imread(image_path)
    mask = np.zeros(img.shape[:2], np.uint8)

    # Create temporary arrays used by GrabCut algorithm
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    # Define the rectangle that contains the object to segment
    height, width = img.shape[:2]
    rect = (10, 10, width - 20, height - 20)

    # Apply GrabCut algorithm
    cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

    # Create a mask where sure background and possible background pixels are set to 0, and the rest to 1
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

    # Apply the mask to the image
    img = img * mask2[:, :, np.newaxis]

    return img

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

def extract_colors(image_path, num_colors=3):
    # Remove the background from the image
    img = remove_background(image_path)

    # Convert image to RGB from BGR (OpenCV uses BGR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Resize image to reduce processing time
    img = cv2.resize(img, (600, 400), interpolation=cv2.INTER_AREA)

    # Reshape image to a list of pixels
    img = img.reshape((img.shape[0] * img.shape[1], 3))

    # Remove black pixels (background)
    img = img[~np.all(img == [0, 0, 0], axis=1)]

    # Check if there are enough pixels left after background removal
    if len(img) == 0:
        return [("#000000", "Black")]

    # Use KMeans to cluster pixels
    kmeans = KMeans(n_clusters=num_colors)
    kmeans.fit(img)

    # Get colors and counts
    counts = Counter(kmeans.labels_)
    center_colors = kmeans.cluster_centers_

    # Sort colors by frequency
    ordered_colors = [center_colors[i] for i in counts.keys()]

    # Convert RGB to Hex
    hex_colors = [rgb_to_hex(ordered_colors[i]) for i in range(len(ordered_colors))]

    # Get color names
    color_names = [get_color_name(hex_colors[i]) for i in range(len(hex_colors))]

    # Return list of tuples (hex_color, color_name)
    return list(zip(hex_colors, color_names))

def rgb_to_hex(color):
    return "#{:02x}{:02x}{:02x}".format(int(color[0]), int(color[1]), int(color[2]))

if __name__ == '__main__':
    app.run(debug=True)
