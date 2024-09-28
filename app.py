from flask import Flask, render_template, request, redirect, url_for
import os
import cv2
import numpy as np
from sklearn.cluster import KMeans
from collections import Counter
from flask_sqlalchemy import SQLAlchemy
import webcolors

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
    colors = db.Column(db.String(500), nullable=False)  # Increased size to accommodate names

    def __repr__(self):
        return f'<ClothingItem {self.id}>'

with app.app_context():
    db.create_all()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return 'No file part', 400

    file = request.files['file']
    if file.filename == '':
        return 'No selected file', 400

    if file:
        filename = file.filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Process the image
        colors = extract_colors(filepath)  # Now returns a list of (hex_color, color_name)

        # Save to database
        # Convert list of tuples to a string for storage, e.g., "hex1,name1;hex2,name2;hex3,name3"
        colors_str = ';'.join([f"{hex_color},{color_name}" for hex_color, color_name in colors])
        new_item = ClothingItem(image_filename=filename, colors=colors_str)
        db.session.add(new_item)
        db.session.commit()

        # Render results
        return render_template('result.html', colors=colors, filename=filename)

@app.route('/wardrobe')
def wardrobe():
    items = ClothingItem.query.all()
    return render_template('wardrobe.html', items=items)

def closest_colour(requested_colour):
    min_colours = {}
    for hex_code, name in webcolors.CSS3_HEX_TO_NAMES.items():
        r_c, g_c, b_c = webcolors.hex_to_rgb(hex_code)
        rd = (r_c - requested_colour.red) ** 2
        gd = (g_c - requested_colour.green) ** 2
        bd = (b_c - requested_colour.blue) ** 2
        min_colours[(rd + gd + bd)] = name
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
    # Load image using OpenCV
    img = cv2.imread(image_path)
    # Convert image to RGB from BGR (OpenCV uses BGR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Resize image to reduce processing time
    img = cv2.resize(img, (600, 400))
    # Reshape image to a list of pixels
    img = img.reshape((img.shape[0] * img.shape[1], 3))
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
