from flask import Flask, render_template, request, redirect, url_for
import os
from PIL import Image
import cv2
import numpy as np
from sklearn.cluster import KMeans
from collections import Counter

app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

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
        colors = extract_colors(filepath)
        # Render results
        return render_template('result.html', colors=colors, filename=filename)

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
    hex_colors = [rgb_to_hex(ordered_colors[i]) for i in range(len(ordered_colors))]
    return hex_colors

def rgb_to_hex(color):
    return "#{:02x}{:02x}{:02x}".format(int(color[0]), int(color[1]), int(color[2]))

if __name__ == '__main__':
    app.run(debug=True)
