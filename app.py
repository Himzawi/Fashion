from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import requests
from dotenv import load_dotenv
import os
import traceback

# Load environment variables
load_dotenv("Api.env")
api_key = os.getenv("OPENROUTER_API_KEY")

# Ensure API key is loaded
if not api_key:
    raise ValueError("OPENROUTER_API_KEY is not set in Api.env")

# Add OpenWeatherMap API key
weather_api_key = os.getenv("OPENWEATHER_API_KEY", "f2055b01a1ae00d86ae74afb0d6e2ec3")

app = Flask(__name__)

# Configure CORS - PROPERLY specifying origins
CORS(app, resources={r"/*": {"origins": ["https://ai-fashion-advisor.web.app", "http://localhost:3000"], 
                             "methods": ["GET", "POST", "OPTIONS"], 
                             "allow_headers": ["Content-Type"]}})

# Load CLIP model and processor
try:
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
except Exception as e:
    print(f"Error loading CLIP model: {e}")
    raise

# Ensure 'uploads' directory exists
os.makedirs('uploads', exist_ok=True)

# Add a root endpoint to handle those 404 errors
@app.route('/', methods=['GET'])
def index():
    return jsonify({"status": "AI Fashion Advisor API is running"}), 200


@app.route('/weather', methods=['GET'])
def get_weather():
    try:
        # Get latitude and longitude from request parameters
        latitude = request.args.get('lat')
        longitude = request.args.get('lon')
        
        if not latitude or not longitude:
            return jsonify({'error': 'Latitude and longitude are required'}), 400
            
        # Call OpenWeatherMap API
        url = f"https://api.openweathermap.org/data/2.5/weather?lat={latitude}&lon={longitude}&appid={weather_api_key}&units=metric"
        response = requests.get(url)
        response.raise_for_status()
        
        # Return the weather data
        return jsonify(response.json())
        
    except Exception as e:
        error_message = f"Error fetching weather data: {str(e)}"
        print(error_message)
        traceback.print_exc()
        return jsonify({'error': error_message}), 500


@app.route('/upload', methods=['POST'])
def upload():
    try:
        print("Received upload request")

        if 'file' not in request.files:
            print("No file uploaded in request")
            return jsonify({'error': 'No file uploaded'}), 400

        file = request.files['file']

        if file.filename == '':
            print("No file selected")
            return jsonify({'error': 'No file selected'}), 400

        # Get weather data if provided
        weather_data = None
        latitude = request.form.get('latitude')
        longitude = request.form.get('longitude')
        
        if latitude and longitude:
            try:
                # Call OpenWeatherMap API
                url = f"https://api.openweathermap.org/data/2.5/weather?lat={latitude}&lon={longitude}&appid={weather_api_key}&units=metric"
                weather_response = requests.get(url)
                weather_response.raise_for_status()
                weather_data = weather_response.json()
            except Exception as weather_error:
                print(f"Error fetching weather data: {weather_error}")
                # Continue even if weather data fetching fails

        # Save the uploaded file
        filename = file.filename
        image_path = os.path.join('uploads', filename)
        file.save(image_path)
        print(f"File saved to: {image_path}")

        # Analyze the outfit
        feedback, outfit_description = analyze_outfit(image_path)
        print(f"Feedback: {feedback}")
        print(f"Outfit description: {outfit_description}")

        # Generate outfit recommendations
        recommendations = generate_suggestions(feedback)
        print(f"Recommendations: {recommendations}")

        # Generate remixing suggestions
        remixing_suggestions = generate_remixing_suggestions(outfit_description)
        print(f"Remixing suggestions: {remixing_suggestions}")

        # Add weather recommendations if weather data is available
        weather_recommendations = ""
        if weather_data:
            weather_recommendations = get_weather_recommendations(weather_data, feedback)
            print(f"Weather recommendations: {weather_recommendations}")

        return jsonify({
            'feedback': feedback,
            'recommendations': recommendations,
            'weather_recommendations': weather_recommendations,
            'remixing_suggestions': remixing_suggestions
        })

    except Exception as e:
        error_message = f"Error in /upload: {str(e)}"
        print(error_message)
        traceback.print_exc()
        return jsonify({'error': error_message}), 500


def get_weather_recommendations(weather_data, feedback):
    if not weather_data:
        return "Unable to provide weather-based recommendations due to a problem fetching weather data."
    
    temperature = weather_data['main']['temp']
    weather_condition = weather_data['weather'][0]['main'].lower()

    recommendations = ''

    # Check temperature
    if temperature < 10:
        recommendations += "It's very cold! "
        if 'shorts' in feedback.lower():  # Fixed includes() to in operator
            recommendations += 'Consider wearing pants instead of shorts. '
        if 'jacket' not in feedback.lower():  # Fixed includes() to in operator
            recommendations += 'You should wear a jacket. '
    elif temperature >= 10 and temperature < 20:
        recommendations += "It's a bit chilly. "
        if 'shorts' in feedback.lower():
            recommendations += 'Consider wearing pants. '
        if 'jacket' not in feedback.lower():
            recommendations += 'A light jacket might be a good idea. '
    elif temperature >= 20:
        recommendations += "It's warm! "
        if 'jacket' in feedback.lower():
            recommendations += 'You might want to take off your jacket. '
        if 'pants' in feedback.lower():
            recommendations += 'Consider wearing shorts. '

    # Check weather conditions
    if 'rain' in weather_condition:
        recommendations += "It's raining. Don't forget an umbrella or a raincoat!"
    elif 'snow' in weather_condition:
        recommendations += "It's snowing. Bundle up and stay warm!"
    elif 'clear' in weather_condition:
        recommendations += 'The weather is clear. Enjoy your day!'

    return recommendations


def analyze_outfit(image_path):
    try:
        image = Image.open(image_path)

        clothing_items = ["suit", "t-shirt", "jeans", "dress", "jacket", "shorts", "skirt", "hoodie", "shirt",
                          "sweater"]
        styles = ["casual", "formal", "sporty", "elegant", "bohemian", "streetwear"]

        inputs_items = processor(text=clothing_items, images=image, return_tensors="pt", padding=True)
        outputs_items = model(**inputs_items)
        logits_per_image_items = outputs_items.logits_per_image
        probs_items = logits_per_image_items.softmax(dim=1).tolist()[0]
        top_items = sorted(zip(clothing_items, probs_items), key=lambda x: x[1], reverse=True)[:3]

        inputs_styles = processor(text=styles, images=image, return_tensors="pt", padding=True)
        outputs_styles = model(**inputs_styles)
        logits_per_image_styles = outputs_styles.logits_per_image
        probs_styles = logits_per_image_styles.softmax(dim=1).tolist()[0]
        top_styles = sorted(zip(styles, probs_styles), key=lambda x: x[1], reverse=True)[:3]

        feedback = f"This outfit is {top_styles[0][0]}! It also works well for {top_styles[1][0]} and {top_styles[2][0]}."
        outfit_description = f"The outfit includes a {top_items[0][0]}, {top_items[1][0]}, and {top_items[2][0]}."

        return feedback, outfit_description

    except Exception as e:
        error_message = f"Error in analyze_outfit: {str(e)}"
        print(error_message)
        traceback.print_exc()
        return "Error analyzing outfit.", "Outfit analysis failed."


def generate_suggestions(style):
    try:
        print("Sending request to DeepSeek-R1 API via OpenRouter...")
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://ai-fashion-advisor.web.app/",
            "X-Title": "Outfit Advisor"
        }
        data = {
            "model": "deepseek/deepseek-r1:free",
            "messages": [
                {
                    "role": "system",
                    "content": "You are a fashion advisor. Provide 3 concise outfit suggestions based on the user's style. For each suggestion, include:\n- **Top**: The top to wear.\n- **Bottom**: The bottom to wear.\n- **Footwear**: Recommended footwear.\n- **Accessories**: Recommended accessories.\nEach suggestion should be a single line, starting with a number and a name (e.g., '1. **Casual Chic**'). Do not include explanations or introductions so only give the bullet points dont speak to yourself."
                },
                {
                    "role": "user",
                    "content": f"Suggest 3 outfits for a {style} look. Keep it short and simple."
                }
            ]
        }

        print("Request Data:", data)
        response = requests.post("https://openrouter.ai/api/v1/chat/completions", json=data, headers=headers)
        response.raise_for_status()
        response_json = response.json()
        print("DeepSeek API Response:", response_json)

        if "choices" in response_json and len(response_json["choices"]) > 0:
            return response_json["choices"][0]["message"]["content"]
        else:
            return f"Error: No 'choices' found in API response: {response_json}"

    except requests.exceptions.RequestException as e:
        error_message = f"Network error during API call: {str(e)}"
        print(error_message)
        return error_message
    except Exception as e:
        error_message = f"Error generating suggestions: {str(e)}"
        print(error_message)
        traceback.print_exc()
        return error_message


def generate_remixing_suggestions(outfit_description):
    try:
        print("Sending request to DeepSeek-R1 API via OpenRouter...")
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://ai-fashion-advisor.web.app/",
            "X-Title": "Outfit Advisor"
        }
        data = {
            "model": "deepseek/deepseek-r1:free",
            "messages": [
                {
                    "role": "system",
                    "content": "You are a fashion advisor. Provide 3 concise and actionable ways to remix the user's outfit. For each suggestion, include:\n- **Swap**: What to change.\n- **Footwear**: Recommended footwear.\n- **Accessories**: Recommended accessories.\nEach suggestion should be a single line, starting with a number and a name (e.g., '1. **Streetwear Edge**'). Do not include explanations or introductions."
                },
                {
                    "role": "user",
                    "content": f"The user is wearing: {outfit_description}. Suggest 3 ways to remix this outfit."
                }
            ]
        }

        print("Request Data:", data)
        response = requests.post("https://openrouter.ai/api/v1/chat/completions", json=data, headers=headers)
        response.raise_for_status()
        response_json = response.json()
        print("DeepSeek API Response:", response_json)

        if "choices" in response_json and len(response_json["choices"]) > 0:
            return response_json["choices"][0]["message"]["content"]
        else:
            return f"Error: No 'choices' found in API response: {response_json}"

    except requests.exceptions.RequestException as e:
        error_message = f"Network error during API call: {str(e)}"
        print(error_message)
        return error_message
    except Exception as e:
        error_message = f"Error generating remixing suggestions: {str(e)}"
        print(error_message)
        traceback.print_exc()
        return error_message


if __name__ == '__main__':
    # Use 0.0.0.0 to make it accessible externally
    app.run(host='0.0.0.0', port=8080, debug=True)