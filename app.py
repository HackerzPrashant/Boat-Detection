import os
import numpy as np
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from werkzeug.utils import secure_filename

app = Flask(__name__)

# --- CONFIGURATION --- lets see now
MODEL_PATH = 'topviwandsonar_transfer_model.h5'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Load Model (Global scope to load only once)
print(f"Loading model from {MODEL_PATH}...")
try:
    model = load_model(MODEL_PATH)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Make sure the .h5 file is in the same folder as app.py")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    
    if file and allowed_file(file.filename):
        # 1. Save temp file (to handle different image formats safely)
        filename = secure_filename(file.filename)
        filepath = os.path.join('static', filename)
        
        # Ensure static dir exists for temp storage
        if not os.path.exists('static'):
            os.makedirs('static')
            
        file.save(filepath)

        # 2. Preprocess Image (Exact same steps as your training)
        # Load image, force RGB, resize to 128x128
        image = load_img(filepath, target_size=(128, 128), color_mode='rgb')
        image_arr = img_to_array(image)
        image_arr = np.expand_dims(image_arr, axis=0) # Shape: (1, 128, 128, 3)

        # 3. Predict
        # Your model has preprocessing baked in, so we pass raw [0-255] values
        prediction_score = model.predict(image_arr)[0][0]
        
        # 4. Cleanup
        os.remove(filepath)

        # 5. Interpret Result
        # Your logic: > 0.5 is Boat, else Background
        if prediction_score > 0.5:
            label = "Suspicious Object (Boat) Detected"
            status = "DANGER"
        else:
            label = "Clear Water / Background"
            status = "SAFE"
            
        return jsonify({
            'label': label,
            'confidence': float(prediction_score),
            'status': status
        })

    return jsonify({'error': 'Invalid file type'}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)