from flask import Flask, render_template, request, jsonify
import os
from werkzeug.utils import secure_filename
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np
import cv2

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load ResNet50 model
model = ResNet50(weights='imagenet', include_top=False)

# Define cancer types and stages
CANCER_TYPES = [
    "Pancreatic Ductal Adenocarcinoma",
    "Neuroendocrine Tumor",
    "Acinar Cell Carcinoma"
]

CANCER_STAGES = [
    "Stage I: Early, localized cancer",
    "Stage II: Locally advanced cancer",
    "Stage III: Cancer has spread to nearby structures",
    "Stage IV: Cancer has spread to distant organs"
]

def process_image_steps(image_path):
    """
    Process image through various steps and save intermediate results
    """
    # Read the original image
    img = cv2.imread(image_path)
    filename = os.path.basename(image_path)
    base_name = os.path.splitext(filename)[0]
    results = {}

    # 1. Original
    results['original'] = f'/static/uploads/{filename}'

    # 2. Threshold image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    thresh_path = f'{app.config["UPLOAD_FOLDER"]}/{base_name}_threshold.png'
    cv2.imwrite(thresh_path, thresh)
    results['threshold'] = f'/static/uploads/{base_name}_threshold.png'

    # 3. Pancreas masking (simulated)
    mask = np.zeros_like(gray)
    center_y, center_x = gray.shape[0] // 2, gray.shape[1] // 2
    cv2.ellipse(mask, (center_x, center_y), (100, 50), 0, 0, 360, 255, -1)
    masked = cv2.bitwise_and(img, img, mask=mask)
    mask_path = f'{app.config["UPLOAD_FOLDER"]}/{base_name}_masked.png'
    cv2.imwrite(mask_path, masked)
    results['masked'] = f'/static/uploads/{base_name}_masked.png'

    # 4. Pancreas segmentation (simulated)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    segmented = img.copy()
    cv2.drawContours(segmented, contours, -1, (0, 255, 0), 2)
    segmented_path = f'{app.config["UPLOAD_FOLDER"]}/{base_name}_segmented.png'
    cv2.imwrite(segmented_path, segmented)
    results['segmented'] = f'/static/uploads/{base_name}_segmented.png'

    # 5. Nodule segmentation (simulated)
    nodules = img.copy()
    # Simulate some nodules
    for _ in range(3):
        x = np.random.randint(center_x - 50, center_x + 50)
        y = np.random.randint(center_y - 25, center_y + 25)
        cv2.circle(nodules, (x, y), 10, (255, 0, 0), -1)
    nodules_path = f'{app.config["UPLOAD_FOLDER"]}/{base_name}_nodules.png'
    cv2.imwrite(nodules_path, nodules)
    results['nodules'] = f'/static/uploads/{base_name}_nodules.png'

    # 6. Cancerous nodules (simulated)
    cancerous = nodules.copy()
    # Mark some nodules as cancerous
    for _ in range(2):
        x = np.random.randint(center_x - 50, center_x + 50)
        y = np.random.randint(center_y - 25, center_y + 25)
        cv2.circle(cancerous, (x, y), 12, (0, 0, 255), 2)
    cancerous_path = f'{app.config["UPLOAD_FOLDER"]}/{base_name}_cancerous.png'
    cv2.imwrite(cancerous_path, cancerous)
    results['cancerous'] = f'/static/uploads/{base_name}_cancerous.png'

    return results

def predict_cancer(image_path):
    """
    Process image and make predictions using ResNet50
    """
    # Load and preprocess image
    img = image.load_img(image_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    # Get features from ResNet50
    features = model.predict(x)

    # Process image through different steps
    processing_results = process_image_steps(image_path)

    # For demo purposes, simulate cancer detection
    has_cancer = np.random.random() > 0.5

    prediction = {
        'hasCancer': has_cancer,
        'confidence': float(np.random.random() * 0.3 + 0.7),
        'type': np.random.choice(CANCER_TYPES) if has_cancer else None,
        'stage': np.random.choice(CANCER_STAGES) if has_cancer else None,
        'regions': [
            {
                'x': float(np.random.random() * 0.8),
                'y': float(np.random.random() * 0.8),
                'width': float(np.random.random() * 0.2 + 0.1),
                'height': float(np.random.random() * 0.2 + 0.1)
            }
        ] if has_cancer else [],
        'processingSteps': processing_results
    }
    return prediction

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Process image and get prediction
        try:
            result = predict_cancer(filepath)
            result['imageUrl'] = f'/static/uploads/{filename}'
            return jsonify(result)
        except Exception as e:
            return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)