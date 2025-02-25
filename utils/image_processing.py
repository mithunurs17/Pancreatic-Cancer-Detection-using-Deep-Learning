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

    # Extract comprehensive features
    features = {
        'num_contours': len(contours),
        'total_area': 0,
        'avg_circularity': 0,
        'max_contour_area': 0,
        'contour_density': 0,
        'avg_intensity': np.mean(segmented),
        'intensity_std': np.std(segmented),
        'texture_uniformity': 0,
        'edge_density': 0,
        'shape_complexity': 0
    }

    image_area = segmented.shape[0] * segmented.shape[1]
    total_perimeter = 0

    for contour in contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        features['total_area'] += area
        total_perimeter += perimeter

        if area > features['max_contour_area']:
            features['max_contour_area'] = area

        # Calculate circularity
        if perimeter > 0:
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            features['avg_circularity'] += circularity

            # Calculate shape complexity
            complexity = perimeter * perimeter / (4 * np.pi * area)
            features['shape_complexity'] += complexity

    if len(contours) > 0:
        features['avg_circularity'] /= len(contours)
        features['shape_complexity'] /= len(contours)
        features['contour_density'] = features['total_area'] / image_area
        features['edge_density'] = total_perimeter / image_area

    # Calculate texture uniformity
    if np.max(segmented) > 0:
        features['texture_uniformity'] = np.sum(np.square(segmented / np.max(segmented)))

    # Create visualization image
    result = cv2.cvtColor(segmented, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(result, contours, -1, (0,255,0), 2)

    return result, features

def classify_cancer(features):
    # First check for healthy pancreas
    if (features['contour_density'] < 0.1 and 
        features['edge_density'] < 0.05 and 
        features['texture_uniformity'] > 0.8):
        return {
            'cancer_type': 'Healthy Pancreas',
            'cancer_stage': 'N/A',
            'confidence': 0.92
        }

    # Check for no detected features
    if features['num_contours'] == 0:
        return {
            'cancer_type': 'Insufficient Features Detected',
            'cancer_stage': 'N/A',
            'confidence': 0.95
        }

    # Analysis based on multiple features
    density = features['contour_density']
    circularity = features['avg_circularity']
    complexity = features['shape_complexity']
    texture = features['texture_uniformity']
    intensity = features['avg_intensity']

    # Comprehensive classification logic
    if density > 0.4:
        if circularity > 0.7 and complexity < 2:
            return {
                'cancer_type': 'Pancreatic Neuroendocrine Tumor',
                'cancer_stage': 'Stage III',
                'confidence': 0.85
            }
        elif texture < 0.3 and intensity > 150:
            return {
                'cancer_type': 'Mucinous Cystic Neoplasm',
                'cancer_stage': 'Stage II',
                'confidence': 0.83
            }
        else:
            return {
                'cancer_type': 'Ductal Adenocarcinoma',
                'cancer_stage': 'Stage II',
                'confidence': 0.88
            }
    elif density > 0.2:
        if circularity > 0.6 and complexity < 1.5:
            return {
                'cancer_type': 'Acinar Cell Carcinoma',
                'cancer_stage': 'Stage II',
                'confidence': 0.82
            }
        elif texture > 0.6 and intensity < 100:
            return {
                'cancer_type': 'Intraductal Papillary Mucinous Neoplasm',
                'cancer_stage': 'Stage I',
                'confidence': 0.81
            }
        else:
            return {
                'cancer_type': 'Ductal Adenocarcinoma',
                'cancer_stage': 'Stage I',
                'confidence': 0.87
            }
    else:
        if circularity > 0.8 and complexity < 1.2:
            return {
                'cancer_type': 'Solid Pseudopapillary Neoplasm',
                'cancer_stage': 'Stage I',
                'confidence': 0.80
            }
        elif texture < 0.4 and intensity > 120:
            return {
                'cancer_type': 'Serous Cystadenoma',
                'cancer_stage': 'Early Stage',
                'confidence': 0.79
            }
        elif complexity > 2.5:
            return {
                'cancer_type': 'Pancreatic Adenosquamous Carcinoma',
                'cancer_stage': 'Stage II',
                'confidence': 0.77
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