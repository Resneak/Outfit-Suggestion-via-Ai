AI-Powered Outfit Recommendation Web Application
This project is an AI-powered outfit recommendation web application that provides suggestions based on your wardrobe, weather conditions, and outfit types. It uses machine learning models to recognize clothing from images and assist users in curating outfits for different occasions.

Live Application
You can view the live version of the app at: http://3.87.59.31:5000/

Features
Image Upload and Processing: Users can upload images of clothing items, and the application removes the background and analyzes the clothing.
Clothing Classification: Uses a ResNet34 model to classify clothing into detailed categories such as tops, bottoms, outerwear, footwear, and accessories.
Outfit Recommendation: Provides personalized outfit suggestions by matching items from different categories, factoring in color harmony.
Color Detection: Uses KMeans clustering to extract the primary colors from each clothing item and categorizes the colors.
Weather-Based Recommendations: Fetches current weather data using the OpenWeather API and suggests appropriate outfits.
Wardrobe Management: Allows users to store, categorize, and filter clothing items in their wardrobe.
Tech Stack
Backend: Flask (Python)
Frontend: HTML, CSS, Jinja2 (for templating)
Database: SQLite (for storing clothing items)
Machine Learning: FastAI, PyTorch, ResNet34
Background Removal: U²-Net (for removing the background from clothing images)
Deployment: AWS Elastic Beanstalk
Weather Data: OpenWeather API
Installation
Prerequisites
Python 3.9 or higher
Flask
SQLite (for local database)
Clone the Repository
bash
Copy code
git clone https://github.com/yourusername/ai-powered-outfit-recommendation.git
cd ai-powered-outfit-recommendation
Install Dependencies
Create a virtual environment and install the dependencies from the requirements.txt file.

bash
Copy code
python3 -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
pip install -r requirements.txt
Set Up Environment Variables
You need to set up your environment variables to include your OpenWeather API key.

bash
Copy code
export OPENWEATHER_API_KEY=your_api_key
Alternatively, you can create a .env file and add your API key like this:

bash
Copy code
OPENWEATHER_API_KEY=your_api_key
Run the Application
Once everything is set up, run the application locally:

bash
Copy code
flask run
The application will be available at http://127.0.0.1:5000/.

Deployment
This project is deployed on AWS Elastic Beanstalk. To deploy, follow the standard Elastic Beanstalk deployment process:

Zip your project files, including application.py, requirements.txt, and Procfile.
Create and configure an Elastic Beanstalk environment.
Upload your zip file to Elastic Beanstalk.
The app should be live on your provided domain.
Project Structure
php
Copy code
.
├── instance/
├── model_training/
├── models/
├── static/
│   └── uploads/              # Image upload folder
├── templates/
│   └── index.html            # Main HTML page for the web app
├── application.py            # Main Flask application
├── u2net.pth                 # Pretrained U²-Net model for background removal
├── u2net.py                  # U²-Net architecture script
├── requirements.txt          # Python dependencies
└── README.md                 # Project README file
Model
U²-Net
U²-Net is used for background removal from uploaded images. The pre-trained model (u2net.pth) is loaded and run on the server to process images.

ResNet34
A FastAI-trained ResNet34 model is used to classify clothing items into categories (e.g., tops, bottoms, outerwear).

API Integration
OpenWeather API: Used to fetch weather data based on user input (latitude and longitude) to recommend weather-appropriate outfits.
Future Enhancements
User Authentication: Add the ability for users to create accounts and manage personalized wardrobes.
Seasonal Recommendations: Enhance recommendations based on seasonal trends and preferences.
Mobile Version: Optimize the app for mobile devices.
Integration with Online Shopping: Add the ability to suggest where users can purchase similar items based on their wardrobe.
Contributing
Contributions are welcome! If you'd like to improve this project or add new features, feel free to open a pull request or raise an issue.

Fork the project
Create your feature branch (git checkout -b feature/your-feature-name)
Commit your changes (git commit -m 'Add some feature')
Push to the branch (git push origin feature/your-feature-name)
Open a pull request
License
This project is licensed under the MIT License - see the LICENSE file for details.
