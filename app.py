from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import requests
from dotenv import load_dotenv
import os
import traceback  # Import traceback for detailed error logging

# Load environment variables (Api.env should be at the same level as your script or specify the full path)
load_dotenv("Api.env")
api_key = os.getenv("OPENROUTER_API_KEY")

# Ensure API key is loaded and raise an exception if it's missing.  This is critical.
if not api_key:
    raise ValueError("OPENROUTER_API_KEY is not set in Api.env")

app = Flask(__name__)

# Configure CORS (restrictive in production)
CORS(app, resources={r"/upload": {"origins": "https://ai-fashion-advisor.web.app"}})  # Correct route

# Load CLIP model and processor (move outside the function for faster loading during startup)
try:
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
except Exception as e:
    print(f"Error loading CLIP model: {e}")
    raise  # Re-raise the exception to halt the app if the model fails to load

# Ensure 'uploads' directory exists (do this only once during startup)
os.makedirs('uploads', exist_ok=True)


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

        # Secure filename (important for security!)
        filename = file.filename  # Replace secure_filename with your own secure method or a library
        image_path = os.path.join('uploads', filename)  # removed secure_filename for now

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

        return jsonify({
            'feedback': feedback,
            'recommendations': recommendations,
            'remixing_suggestions': remixing_suggestions
        })

    except Exception as e:
        error_message = f"Error in /upload: {str(e)}"
        print(error_message)
        traceback.print_exc()  # Print the full traceback
        return jsonify({'error': error_message}), 500


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
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
        response_json = response.json()
        print("DeepSeek API Response:", response_json)

        if "choices" in response_json and len(response_json["choices"]) > 0:  # Check if choices are available
            return response_json["choices"][0]["message"]["content"]
        else:
            return f"Error: No 'choices' found in API response: {response_json}" # More specific error

    except requests.exceptions.RequestException as e: # Catch request exceptions (network issues etc.)
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
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
        response_json = response.json()
        print("DeepSeek API Response:", response_json)

        if "choices" in response_json and len(response_json["choices"]) > 0:  # Check if choices are available
            return response_json["choices"][0]["message"]["content"]
        else:
            return f"Error: No 'choices' found in API response: {response_json}" # More specific error

    except requests.exceptions.RequestException as e: # Catch request exceptions (network issues etc.)
        error_message = f"Network error during API call: {str(e)}"
        print(error_message)
        return error_message
    except Exception as e:
        error_message = f"Error generating remixing suggestions: {str(e)}"
        print(error_message)
        traceback.print_exc()
        return error_message


if __name__ == '__main__':
    app.run(debug=True)