from flask import Flask, render_template, request, jsonify
import numpy as np
import cv2
import base64
from io import BytesIO
from PIL import Image
import os
import sys

app = Flask(__name__)

# Initialize model as None
MODEL = None

def load_model_safe():
    """Load model safely with error handling"""
    global MODEL
    if MODEL is None:
        try:
            # Import tensorflow here to avoid issues
            from tensorflow.keras.models import load_model
            
            # Try different possible paths for the model
            model_paths = [
                "recognizer_model.keras",
                "./recognizer_model.keras",
                os.path.join(os.path.dirname(__file__), "recognizer_model.keras")
            ]
            
            for path in model_paths:
                if os.path.exists(path):
                    MODEL = load_model(path)
                    print(f"Model loaded successfully from: {path}")
                    return True
            
            print("Error: Could not find model file in any expected location")
            return False
            
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    return True

LABELS = {0: "Zero", 1: "One", 2: "Two", 3: "Three", 4: "Four", 
          5: "Five", 6: "Six", 7: "Seven", 8: "Eight", 9: "Nine"}

IMAGESAVE = False  # Disabled for production

def preprocess_for_mnist(img_array):
    """
    Advanced preprocessing to match MNIST format exactly
    """
    if img_array.size == 0:
        return None
    
    # Convert to grayscale if needed
    if len(img_array.shape) == 3:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    
    # Normalize to 0-255 range
    img_array = img_array.astype(np.uint8)
    
    # Invert if needed (MNIST has white digits on black background)
    if np.mean(img_array) > 127:
        img_array = 255 - img_array
    
    # Remove noise - only keep pixels above threshold
    img_array[img_array < 50] = 0
    
    # Find the bounding box of the digit
    coords = cv2.findNonZero(img_array)
    if coords is None:
        return None
    
    x, y, w, h = cv2.boundingRect(coords)
    
    # Add some padding
    padding = max(w, h) // 10
    x = max(0, x - padding)
    y = max(0, y - padding)
    w = min(img_array.shape[1] - x, w + 2 * padding)
    h = min(img_array.shape[0] - y, h + 2 * padding)
    
    # Crop to bounding box
    cropped = img_array[y:y+h, x:x+w]
    
    if cropped.size == 0:
        return None
    
    # Make it square by padding
    max_dim = max(cropped.shape)
    square_img = np.zeros((max_dim, max_dim), dtype=np.uint8)
    
    # Center the digit
    y_offset = (max_dim - cropped.shape[0]) // 2
    x_offset = (max_dim - cropped.shape[1]) // 2
    square_img[y_offset:y_offset+cropped.shape[0], x_offset:x_offset+cropped.shape[1]] = cropped
    
    # Resize to 20x20
    resized = cv2.resize(square_img, (20, 20), interpolation=cv2.INTER_AREA)
    
    # Add 4-pixel border to make 28x28 (like MNIST)
    final_img = np.zeros((28, 28), dtype=np.uint8)
    final_img[4:24, 4:24] = resized
    
    # Apply Gaussian blur to smooth
    final_img = cv2.GaussianBlur(final_img, (1, 1), 0)
    
    # Normalize to 0-1 range
    final_img = final_img.astype(np.float32) / 255.0
    
    return final_img

def get_prediction_with_confidence(image):
    """
    Get prediction with detailed confidence analysis
    """
    if image is None or MODEL is None:
        return None, 0, [], []
    
    try:
        predictions = MODEL.predict(image.reshape(1, 28, 28, 1), verbose=0)
        predicted_class = np.argmax(predictions)
        confidence = np.max(predictions) * 100
        
        # Get all predictions sorted
        all_indices = np.argsort(predictions[0])[::-1]
        all_probs = predictions[0][all_indices] * 100
        
        return predicted_class, confidence, all_indices[:5], all_probs[:5]
    except Exception as e:
        print(f"Prediction error: {e}")
        return None, 0, [], []

def decode_canvas_image(image_data):
    """
    Decode base64 canvas image data to numpy array
    """
    try:
        # Remove data URL prefix
        image_data = image_data.split(',')[1]
        
        # Decode base64
        image_bytes = base64.b64decode(image_data)
        
        # Convert to PIL Image
        image = Image.open(BytesIO(image_bytes))
        
        # Convert to numpy array
        img_array = np.array(image)
        
        return img_array
    except Exception as e:
        print(f"Error decoding image: {e}")
        return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Ensure model is loaded
        if not load_model_safe():
            return jsonify({'error': 'Model not available'}), 500
        
        data = request.get_json()
        image_data = data.get('image')
        
        if not image_data:
            return jsonify({'error': 'No image data provided'}), 400
        
        # Decode the canvas image
        img_array = decode_canvas_image(image_data)
        
        if img_array is None:
            return jsonify({'error': 'Could not decode image'}), 400
        
        # Preprocess for MNIST
        processed_img = preprocess_for_mnist(img_array)
        
        if processed_img is None:
            return jsonify({
                'prediction': None,
                'confidence': 0,
                'message': 'No digit detected',
                'top_predictions': []
            })
        
        # Get prediction
        pred_class, confidence, top_indices, top_probs = get_prediction_with_confidence(processed_img)
        
        if pred_class is None:
            return jsonify({
                'prediction': None,
                'confidence': 0,
                'message': 'Prediction failed',
                'top_predictions': []
            })
        
        # Format top predictions
        top_predictions = []
        for idx, prob in zip(top_indices, top_probs):
            top_predictions.append({
                'digit': LABELS[idx],
                'probability': round(prob, 1)
            })
        
        return jsonify({
            'prediction': LABELS[pred_class],
            'confidence': round(confidence, 1),
            'message': 'Success',
            'top_predictions': top_predictions
        })
        
    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/clear', methods=['POST'])
def clear():
    return jsonify({'message': 'Canvas cleared'})

@app.route('/toggle_save', methods=['POST'])
def toggle_save():
    global IMAGESAVE
    IMAGESAVE = not IMAGESAVE
    return jsonify({
        'save_enabled': IMAGESAVE,
        'message': f"Image saving: {'ON' if IMAGESAVE else 'OFF'}"
    })

@app.route('/health')
def health():
    """Health check endpoint"""
    model_status = "loaded" if MODEL is not None else "not loaded"
    return jsonify({
        'status': 'healthy',
        'model_status': model_status
    })

if __name__ == '__main__':
    # Load model on startup
    load_model_safe()
    
    # Get port from environment variable (required for Render)
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)