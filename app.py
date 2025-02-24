import os
import logging
from flask import Flask, render_template, request, jsonify
from utils.image_processing import process_image
import cv2
import numpy as np
from werkzeug.utils import secure_filename
import base64

# Configure logging
logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET")

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    logging.debug("Rendering index page")
    return render_template('index.html')

@app.route('/about')
def about():
    logging.debug("Rendering about page")
    return render_template('about.html')

@app.route('/process')
def process():
    logging.debug("Rendering process page")
    return render_template('process.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    logging.debug("Received image analysis request")
    if 'image' not in request.files:
        logging.error("No image file in request")
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    if file.filename == '' or not allowed_file(file.filename):
        logging.error("Invalid file or filename")
        return jsonify({'error': 'Invalid file type'}), 400

    try:
        # Read image file
        img_stream = file.read()
        nparr = np.frombuffer(img_stream, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            logging.error("Failed to decode image")
            return jsonify({'error': 'Failed to decode image'}), 400

        # Process image
        results = process_image(img)

        # Convert processed images to base64
        processed_images = {}
        for step, image in results['steps'].items():
            _, buffer = cv2.imencode('.jpg', image)
            img_str = base64.b64encode(buffer).decode('utf-8')
            processed_images[step] = f'data:image/jpeg;base64,{img_str}'

        return jsonify({
            'success': True,
            'processed_images': processed_images,
            'cancer_type': results['cancer_type'],
            'cancer_stage': results['cancer_stage'],
            'confidence': results['confidence']
        })

    except Exception as e:
        logging.error(f"Error processing image: {str(e)}")
        return jsonify({'error': 'Error processing image'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)