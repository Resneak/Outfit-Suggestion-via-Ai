
# AI-Powered Outfit Suggestion Web Application

This project is an AI-powered web application that suggests outfits based on a user's wardrobe, weather conditions, and outfit preferences. The application uses deep learning models for clothing classification and background removal, and it is deployed on AWS Elastic Beanstalk.

## Features

- Upload images of your clothes and categorize them (Top, Bottom, Outerwear, etc.).
- Automatically remove backgrounds from clothing images.
- Predict the clothing category (Top, Bottom, Outerwear, etc.) using a deep learning model.
- Create outfits by selecting clothes from your virtual wardrobe.
- Get outfit suggestions based on weather data (integrated with OpenWeather API).
- Store your wardrobe in a SQLite database.
  
## Technologies Used

- **Flask**: For building the web application.
- **FastAI**: For clothing classification using a ResNet34 model.
- **U²-Net**: For background removal.
- **SQLite**: For storing the uploaded wardrobe items.
- **OpenWeather API**: For fetching weather data.
- **AWS Elastic Beanstalk**: For hosting the application.

## Setup Instructions

### Prerequisites

1. Install [Python 3.10+](https://www.python.org/downloads/).
2. Install [pip](https://pip.pypa.io/en/stable/installation/).
3. Set up an account with [OpenWeather API](https://openweathermap.org/api) and obtain an API key.

### Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/yourusername/ai-outfit.git
    ```

2. Change directory to the project folder:

    ```bash
    cd ai-outfit
    ```

3. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

4. Set up environment variables for the OpenWeather API:

    ```bash
    export OPENWEATHER_API_KEY=your_api_key
    ```

### Running the Application

1. Run the Flask development server:

    ```bash
    python application.py
    ```

2. Access the application at:

    ```
    http://127.0.0.1:5000/
    ```


### Usage

- Navigate to the upload page to add images of clothing items.
- Use the outfit creation tool to mix and match items from your wardrobe.
- Check the weather-based outfit suggestions for daily use.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [FastAI](https://www.fast.ai/) for the deep learning models.
- [U²-Net](https://github.com/xuebinqin/U-2-Net) for background removal.
- [OpenWeather API](https://openweathermap.org/) for weather data.

