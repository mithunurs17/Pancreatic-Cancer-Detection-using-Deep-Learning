import cv2
import numpy as np

def preprocess_image(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Apply histogram equalization
    equalized = cv2.equalizeHist(blurred)
    return equalized

def segment_image(preprocessed):
    # Apply Otsu's thresholding
    _, binary = cv2.threshold(preprocessed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Apply morphological operations
    kernel = np.ones((5,5), np.uint8)
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    return opening

def extract_features(segmented):
    # Find contours
    contours, _ = cv2.findContours(segmented, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Create visualization image
    result = cv2.cvtColor(segmented, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(result, contours, -1, (0,255,0), 2)
    return result, contours

def classify_cancer(features):
    # Dummy classification logic
    # In real implementation, this would use a trained ML model
    return {
        'cancer_type': 'Ductal Adenocarcinoma',
        'cancer_stage': 'Stage II',
        'confidence': 0.85
    }

def process_image(image):
    # Create copies for visualization
    preprocessed = preprocess_image(image.copy())
    segmented = segment_image(preprocessed.copy())
    feature_vis, features = extract_features(segmented.copy())
    
    # Get classification results
    classification = classify_cancer(features)
    
    return {
        'steps': {
            'original': image,
            'preprocessed': cv2.cvtColor(preprocessed, cv2.COLOR_GRAY2BGR),
            'segmented': cv2.cvtColor(segmented, cv2.COLOR_GRAY2BGR),
            'features': feature_vis
        },
        'cancer_type': classification['cancer_type'],
        'cancer_stage': classification['cancer_stage'],
        'confidence': classification['confidence']
    }
