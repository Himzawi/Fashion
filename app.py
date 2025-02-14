from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import requests  # Use requests for OpenRouter API calls
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv("Api.env")
api_key = os.getenv("OPENROUTER_API_KEY")  # Load OpenRouter API key

# Ensure API key is provided
if not api_key:
    raise ValueError("OPENROUTER_API_KEY is not set in Api.env")

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load CLIP model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Ensure "uploads" folder exists
os.makedirs('uploads', exist_ok=True)

@app.route('/upload', methods=['POST'])
def upload():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400

        file = request.files['file']
        image_path = os.path.join('uploads', file.filename)
        file.save(image_path)

        # Analyze the outfit using CLIP
        feedback, outfit_description = analyze_outfit(image_path)

        # Generate outfit recommendations based on the top style
        recommendations = generate_suggestions(feedback)

        # Generate remixing suggestions
        remixing_suggestions = generate_remixing_suggestions(outfit_description)

        return jsonify({
            'feedback': feedback,
            'recommendations': recommendations,
            'remixing_suggestions': remixing_suggestions
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

def analyze_outfit(image_path):
    image = Image.open(image_path)
    
    # Define possible clothing items and styles
    clothing_items = ["suit", "t-shirt", "jeans", "dress", "jacket", "shorts", "skirt", "hoodie", "shirt", "sweater"]
    styles = ["casual", "formal", "sporty", "elegant", "bohemian", "streetwear"]

    # Analyze clothing items
    inputs_items = processor(text=clothing_items, images=image, return_tensors="pt", padding=True)
    outputs_items = model(**inputs_items)
    logits_per_image_items = outputs_items.logits_per_image
    probs_items = logits_per_image_items.softmax(dim=1).tolist()[0]
    top_items = sorted(zip(clothing_items, probs_items), key=lambda x: x[1], reverse=True)[:3]

    # Analyze styles
    inputs_styles = processor(text=styles, images=image, return_tensors="pt", padding=True)
    outputs_styles = model(**inputs_styles)
    logits_per_image_styles = outputs_styles.logits_per_image
    probs_styles = logits_per_image_styles.softmax(dim=1).tolist()[0]
    top_styles = sorted(zip(styles, probs_styles), key=lambda x: x[1], reverse=True)[:3]

    # Generate feedback
    feedback = f"This outfit is {top_styles[0][0]}! It also works well for {top_styles[1][0]} and {top_styles[2][0]}."
    outfit_description = f"The outfit includes a {top_items[0][0]}, {top_items[1][0]}, and {top_items[2][0]}."

    return feedback, outfit_description

def generate_suggestions(style):
    try:
        print("Sending request to DeepSeek-R1 API via OpenRouter...")  # Debugging
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://your-website.com",  # Replace with your website
            "X-Title": "Outfit Advisor"  # Project title
        }
        data = {
            "model": "deepseek/deepseek-r1:free",  # Use DeepSeek-R1 model
            "messages": [
                {
                    "role": "system",
                    "content": "You are a fashion advisor. Provide 3 concise outfit suggestions based on the user's style. For each suggestion, include:\n- **Top**: The top to wear.\n- **Bottom**: The bottom to wear.\n- **Footwear**: Recommended footwear.\n- **Accessories**: Recommended accessories.\nEach suggestion should be a single line, starting with a number and a name (e.g., '1. **Casual Chic**'). Do not include explanations or introductions."
                },
                {
                    "role": "user",
                    "content": f"Suggest 3 outfits for a {style} look. Keep it short and simple."
                }
            ]
        }
        
        print("Request Data:", data)  # Debugging
        response = requests.post("https://openrouter.ai/api/v1/chat/completions", json=data, headers=headers)
        response_json = response.json()
        print("DeepSeek API Response:", response_json)  # Debugging
        
        if response.status_code == 200 and "choices" in response_json:
            return response_json["choices"][0]["message"]["content"]
        else:
            return f"Error: {response_json}"
    except Exception as e:
        print("Error generating suggestions:", str(e))  # Debugging
        return f"Error generating suggestions: {str(e)}"

def generate_remixing_suggestions(outfit_description):
    try:
        print("Sending request to DeepSeek-R1 API via OpenRouter...")  # Debugging
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://your-website.com",  # Replace with your website
            "X-Title": "Outfit Advisor"  # Project title
        }
        data = {
            "model": "deepseek/deepseek-r1:free",  # Use DeepSeek-R1 model
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
        
        print("Request Data:", data)  # Debugging
        response = requests.post("https://openrouter.ai/api/v1/chat/completions", json=data, headers=headers)
        response_json = response.json()
        print("DeepSeek API Response:", response_json)  # Debugging
        
        if response.status_code == 200 and "choices" in response_json:
            return response_json["choices"][0]["message"]["content"]
        else:
            return f"Error: {response_json}"
    except Exception as e:
        print("Error generating remixing suggestions:", str(e))  # Debugging
        return f"Error generating remixing suggestions: {str(e)}"
if __name__ == '__main__':
    app.run(debug=False)