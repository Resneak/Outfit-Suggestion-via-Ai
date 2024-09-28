from flask import Flask, render_template, request, redirect, url_for
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

class ClothingItem(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    image_filename = db.Column(db.String(100), nullable=False)
    colors = db.Column(db.String(500), nullable=False)
    category = db.Column(db.String(50), nullable=False)  # New field for category

    def __repr__(self):
        return f'<ClothingItem {self.id}>'


with app.app_context():
    db.create_all()

# -+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+ ROUTES

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files or 'category' not in request.form:
        return 'Missing file or category', 400

    file = request.files['file']
    category = request.form['category']

    if file.filename == '':
        return 'No selected file', 400

    if category == '':
        return 'No category selected', 400

    if file:
        filename = file.filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Process the image
        colors = extract_colors(filepath)

        # Save to database
        colors_str = ';'.join([f"{hex_color},{color_name}" for hex_color, color_name in colors])
        new_item = ClothingItem(image_filename=filename, colors=colors_str, category=category)
        db.session.add(new_item)
        db.session.commit()

        # Render results
        return render_template('result.html', colors=colors, filename=filename)


@app.route('/wardrobe')
def wardrobe():
    items = ClothingItem.query.all()
    return render_template('wardrobe.html', items=items)

@app.route('/suggestions')
def suggestions():
    suggestions = suggest_outfits()
    return render_template('suggestions.html', suggestions=suggestions)


# -+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+ FUNCTIONS

def get_lab_color(hex_color):
    rgb = sRGBColor.new_from_rgb_hex(hex_color)
    lab = convert_color(rgb, LabColor)
    return lab

def calculate_color_difference(color1, color2):
    lab1 = get_lab_color(color1)
    lab2 = get_lab_color(color2)
    delta_e = delta_e_cie2000(lab1, lab2)
    return delta_e

def suggest_outfits():
    # Get all items from the database
    items = ClothingItem.query.all()

    # Separate items by category
    categories = {
        'Top': [],
        'Bottom': [],
        'Footwear': [],
        'Outerwear': [],
        'Accessory': []
    }
    for item in items:
        if item.category in categories:
            categories[item.category].append(item)

    suggestions = []

    # Ensure we have at least one item in each mandatory category
    if not categories['Top'] or not categories['Bottom'] or not categories['Footwear']:
        return suggestions  # Return empty list if any mandatory category is empty

    # Build outfits with all combinations of Top, Bottom, and Footwear
    for top in categories['Top']:
        top_colors = [pair.split(',')[0] for pair in top.colors.split(';') if pair]
        if not top_colors:
            continue
        top_main_color = top_colors[0]

        for bottom in categories['Bottom']:
            bottom_colors = [pair.split(',')[0] for pair in bottom.colors.split(';') if pair]
            if not bottom_colors:
                continue
            bottom_main_color = bottom_colors[0]

            # Calculate color difference between top and bottom
            color_diff_tb = calculate_color_difference(top_main_color, bottom_main_color)

            for footwear in categories['Footwear']:
                footwear_colors = [pair.split(',')[0] for pair in footwear.colors.split(';') if pair]
                if not footwear_colors:
                    continue
                footwear_main_color = footwear_colors[0]

                # Calculate color difference between bottom and footwear
                color_diff_bf = calculate_color_difference(bottom_main_color, footwear_main_color)

                # Average color difference
                avg_color_diff = (color_diff_tb + color_diff_bf) / 2

                # If average color difference is within a desirable range, consider it a good match
                # Adjust the thresholds as needed
                if 20 < avg_color_diff < 80:
                    outfit = {
                        'top': top,
                        'bottom': bottom,
                        'footwear': footwear,
                        'score': avg_color_diff
                    }

                    # Optionally add outerwear
                    if categories['Outerwear']:
                        outfit['outerwear'] = categories['Outerwear'][0]  # You can implement selection logic here

                    # Optionally add accessory
                    if categories['Accessory']:
                        outfit['accessory'] = categories['Accessory'][0]  # You can implement selection logic here

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
