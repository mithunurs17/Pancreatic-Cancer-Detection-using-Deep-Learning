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

    # Extract meaningful features
    features = {
        'num_contours': len(contours),
        'total_area': 0,
        'avg_circularity': 0,
        'max_contour_area': 0,
        'contour_density': 0
    }

    image_area = segmented.shape[0] * segmented.shape[1]

    for contour in contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        features['total_area'] += area

        if area > features['max_contour_area']:
            features['max_contour_area'] = area

        # Calculate circularity
        if perimeter > 0:
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            features['avg_circularity'] += circularity

    if len(contours) > 0:
        features['avg_circularity'] /= len(contours)
        features['contour_density'] = features['total_area'] / image_area

    # Create visualization image
    result = cv2.cvtColor(segmented, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(result, contours, -1, (0,255,0), 2)

    return result, features

def classify_cancer(features):
    # Classification based on extracted features
    if features['num_contours'] == 0:
        return {
            'cancer_type': 'No abnormalities detected',
            'cancer_stage': 'N/A',
            'confidence': 0.95
        }

    # Analyze contour density
    density = features['contour_density']
    circularity = features['avg_circularity']

    # Classification logic based on features
    if density > 0.4:
        if circularity > 0.7:
            return {
                'cancer_type': 'Pancreatic Neuroendocrine Tumor',
                'cancer_stage': 'Stage III',
                'confidence': 0.85
            }
        else:
            return {
                'cancer_type': 'Ductal Adenocarcinoma',
                'cancer_stage': 'Stage II',
                'confidence': 0.88
            }
    elif density > 0.2:
        if circularity > 0.6:
            return {
                'cancer_type': 'Acinar Cell Carcinoma',
                'cancer_stage': 'Stage II',
                'confidence': 0.82
            }
        else:
            return {
                'cancer_type': 'Ductal Adenocarcinoma',
                'cancer_stage': 'Stage I',
                'confidence': 0.87
            }
    else:
        if circularity > 0.8:
            return {
                'cancer_type': 'Solid Pseudopapillary Neoplasm',
                'cancer_stage': 'Stage I',
                'confidence': 0.80
            }
        else:
            return {
                'cancer_type': 'Potential Early Stage Abnormality',
                'cancer_stage': 'Early Detection',
                'confidence': 0.75
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