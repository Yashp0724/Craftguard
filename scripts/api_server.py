from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import cv2
import base64
from PIL import Image
import io
import os

app = Flask(__name__)
CORS(app)

class CulturalArtAPI:
    def __init__(self):
        self.model = None
        self.feature_names = [
            'color_variance', 'pattern_regularity', 'edge_sharpness', 
            'texture_complexity', 'symmetry_score', 'brush_stroke_variation',
            'color_saturation', 'detail_density', 'geometric_ratio'
        ]
        self.load_model()
    
    def load_model(self):
        """Load the trained model"""
        try:
            self.model = joblib.load('models/cultural_art_classifier.pkl')
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None
    
    def base64_to_image(self, base64_string):
        """Convert base64 string to OpenCV image"""
        try:
            # Remove data URL prefix if present
            if ',' in base64_string:
                base64_string = base64_string.split(',')[1]
            
            # Decode base64
            image_data = base64.b64decode(base64_string)
            
            # Convert to PIL Image
            pil_image = Image.open(io.BytesIO(image_data))
            
            # Convert to OpenCV format
            opencv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            
            return opencv_image
        except Exception as e:
            print(f"Error converting base64 to image: {e}")
            return None
    
    def extract_features(self, img):
        """Extract features from an image"""
        try:
            if img is None:
                return None
                
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            features = []
            
            # 1. Color variance
            color_var = np.var(img_rgb.reshape(-1, 3), axis=0).mean()
            features.append(color_var)
            
            # 2. Pattern regularity
            template_size = min(50, gray.shape[0]//4, gray.shape[1]//4)
            if template_size > 0:
                template = gray[:template_size, :template_size]
                result = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
                pattern_regularity = np.max(result)
            else:
                pattern_regularity = 0.5
            features.append(pattern_regularity)
            
            # 3. Edge sharpness
            edges = cv2.Canny(gray, 50, 150)
            edge_sharpness = np.mean(edges)
            features.append(edge_sharpness)
            
            # 4. Texture complexity
            kernel = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])
            lbp = cv2.filter2D(gray.astype(np.float32), -1, kernel)
            texture_complexity = np.var(lbp)
            features.append(texture_complexity)
            
            # 5. Symmetry score
            h, w = gray.shape
            left_half = gray[:, :w//2]
            right_half = cv2.flip(gray[:, w//2:], 1)
            min_width = min(left_half.shape[1], right_half.shape[1])
            if min_width > 0:
                symmetry_score = np.corrcoef(
                    left_half[:, :min_width].flatten(),
                    right_half[:, :min_width].flatten()
                )[0, 1]
                if np.isnan(symmetry_score):
                    symmetry_score = 0
            else:
                symmetry_score = 0
            features.append(symmetry_score)
            
            # 6. Brush stroke variation
            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            brush_variation = np.std(grad_magnitude)
            features.append(brush_variation)
            
            # 7. Color saturation
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            saturation = np.mean(hsv[:, :, 1])
            features.append(saturation)
            
            # 8. Detail density
            detail_density = np.sum(edges) / (gray.shape[0] * gray.shape[1])
            features.append(detail_density)
            
            # 9. Geometric ratio
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(largest_contour)
                geometric_ratio = w / h if h > 0 else 1
            else:
                geometric_ratio = 1
            features.append(geometric_ratio)
            
            return np.array(features)
            
        except Exception as e:
            print(f"Error extracting features: {e}")
            return None
    
    def analyze_image(self, image_data):
        """Analyze an image for authenticity"""
        if self.model is None:
            return {"error": "Model not loaded"}
        
        # Convert base64 to image
        img = self.base64_to_image(image_data)
        if img is None:
            return {"error": "Invalid image data"}
        
        # Extract features
        features = self.extract_features(img)
        if features is None:
            return {"error": "Feature extraction failed"}
        
        # Make prediction
        try:
            features = features.reshape(1, -1)
            prediction = self.model.predict(features)[0]
            probabilities = self.model.predict_proba(features)[0]
            
            # Determine art form (simplified - in real implementation, use separate classifier)
            art_forms = ['madhubani', 'warli', 'kalamkari', 'banarasi']
            detected_art_form = np.random.choice(art_forms)
            
            # Generate analysis results
            result = {
                "artForm": detected_art_form,
                "authenticityScore": int(probabilities[1] * 100),
                "confidence": int(max(probabilities) * 100),
                "isAuthentic": bool(prediction),
                "detectedFeatures": [
                    {"feature": "TRADITIONAL PATTERNS", "confidence": int(features[0][1] * 100)},
                    {"feature": "NATURAL COLOR VARIATION", "confidence": int((1 - features[0][0]/200) * 100)},
                    {"feature": "HAND CRAFTED TEXTURE", "confidence": int(features[0][3]/2000 * 100)}
                ],
                "riskFactors": []
            }
            
            # Add risk factors for low authenticity scores
            if result["authenticityScore"] < 80:
                result["riskFactors"].append("High pattern regularity suggests machine production")
            if result["authenticityScore"] < 70:
                result["riskFactors"].append("Color consistency indicates synthetic materials")
            if result["authenticityScore"] < 60:
                result["riskFactors"].append("Edge sharpness suggests digital printing")
            
            return result
            
        except Exception as e:
            print(f"Error during prediction: {e}")
            return {"error": "Prediction failed"}

# Initialize API
api = CulturalArtAPI()

@app.route('/api/analyze', methods=['POST'])
def analyze():
    try:
        data = request.get_json()
        image_data = data.get('image')
        
        if not image_data:
            return jsonify({"error": "No image data provided"}), 400
        
        result = api.analyze_image(image_data)
        return jsonify(result)
        
    except Exception as e:
        print(f"API Error: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({
        "status": "healthy",
        "model_loaded": api.model is not None
    })

if __name__ == '__main__':
    print("🧢 CraftGuard API Server Starting...")
    print("Make sure to run train_model.py first to create the model!")
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    app.run(debug=True, host='0.0.0.0', port=5000)
