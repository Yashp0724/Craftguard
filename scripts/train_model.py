import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import cv2
import os
from PIL import Image
import matplotlib.pyplot as plt

class CulturalArtClassifier:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.feature_names = [
            'color_variance', 'pattern_regularity', 'edge_sharpness', 
            'texture_complexity', 'symmetry_score', 'brush_stroke_variation',
            'color_saturation', 'detail_density', 'geometric_ratio'
        ]
        
    def extract_features(self, image_path):
        """Extract features from an image for cultural art classification"""
        try:
            # Load image
            img = cv2.imread(image_path)
            if img is None:
                return None
                
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            features = []
            
            # 1. Color variance (authentic art has more natural color variation)
            color_var = np.var(img_rgb.reshape(-1, 3), axis=0).mean()
            features.append(color_var)
            
            # 2. Pattern regularity (machine-made has more regular patterns)
            # Use template matching to detect repetitive patterns
            template_size = min(50, gray.shape[0]//4, gray.shape[1]//4)
            template = gray[:template_size, :template_size]
            result = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
            pattern_regularity = np.max(result)
            features.append(pattern_regularity)
            
            # 3. Edge sharpness (digital/printed images have sharper edges)
            edges = cv2.Canny(gray, 50, 150)
            edge_sharpness = np.mean(edges)
            features.append(edge_sharpness)
            
            # 4. Texture complexity using Local Binary Pattern
            def calculate_lbp_variance(image):
                # Simplified LBP calculation
                kernel = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])
                lbp = cv2.filter2D(image.astype(np.float32), -1, kernel)
                return np.var(lbp)
            
            texture_complexity = calculate_lbp_variance(gray)
            features.append(texture_complexity)
            
            # 5. Symmetry score (handmade art often has slight asymmetries)
            h, w = gray.shape
            left_half = gray[:, :w//2]
            right_half = cv2.flip(gray[:, w//2:], 1)
            min_width = min(left_half.shape[1], right_half.shape[1])
            symmetry_score = np.corrcoef(
                left_half[:, :min_width].flatten(),
                right_half[:, :min_width].flatten()
            )[0, 1]
            features.append(symmetry_score if not np.isnan(symmetry_score) else 0)
            
            # 6. Brush stroke variation (authentic art has more variation)
            # Calculate gradient magnitude variation
            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            brush_variation = np.std(grad_magnitude)
            features.append(brush_variation)
            
            # 7. Color saturation (natural dyes vs synthetic)
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            saturation = np.mean(hsv[:, :, 1])
            features.append(saturation)
            
            # 8. Detail density
            detail_density = np.sum(edges) / (gray.shape[0] * gray.shape[1])
            features.append(detail_density)
            
            # 9. Geometric ratio (cultural art often follows specific proportions)
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
            print(f"Error processing {image_path}: {e}")
            return None
    
    def create_sample_dataset(self):
        """Create a sample dataset for training"""
        print("Creating sample dataset...")
        
        # Sample data representing different cultural art characteristics
        # In real implementation, you would load actual images
        np.random.seed(42)
        
        # Authentic art samples (higher variation, lower regularity)
        authentic_samples = []
        for i in range(200):
            features = [
                np.random.normal(150, 50),    # color_variance (higher for authentic)
                np.random.normal(0.3, 0.1),  # pattern_regularity (lower for authentic)
                np.random.normal(20, 10),    # edge_sharpness (lower for authentic)
                np.random.normal(1000, 300), # texture_complexity (higher for authentic)
                np.random.normal(0.7, 0.15), # symmetry_score (lower for authentic)
                np.random.normal(80, 20),    # brush_stroke_variation (higher)
                np.random.normal(120, 30),   # color_saturation (moderate)
                np.random.normal(0.15, 0.05), # detail_density
                np.random.normal(1.2, 0.3)   # geometric_ratio
            ]
            authentic_samples.append(features + [1])  # 1 = authentic
        
        # Fake/machine-made samples (lower variation, higher regularity)
        fake_samples = []
        for i in range(200):
            features = [
                np.random.normal(80, 20),     # color_variance (lower for fake)
                np.random.normal(0.8, 0.1),  # pattern_regularity (higher for fake)
                np.random.normal(40, 15),    # edge_sharpness (higher for fake)
                np.random.normal(500, 150),  # texture_complexity (lower for fake)
                np.random.normal(0.9, 0.05), # symmetry_score (higher for fake)
                np.random.normal(30, 10),    # brush_stroke_variation (lower)
                np.random.normal(180, 25),   # color_saturation (higher)
                np.random.normal(0.25, 0.08), # detail_density
                np.random.normal(1.0, 0.1)   # geometric_ratio
            ]
            fake_samples.append(features + [0])  # 0 = fake
        
        # Combine datasets
        all_samples = authentic_samples + fake_samples
        df = pd.DataFrame(all_samples, columns=self.feature_names + ['authentic'])
        
        return df
    
    def train(self, df):
        """Train the classifier"""
        print("Training the model...")
        
        X = df[self.feature_names]
        y = df['authentic']
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train the model
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Model Accuracy: {accuracy:.3f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Fake', 'Authentic']))
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nFeature Importance:")
        print(feature_importance)
        
        return accuracy
    
    def predict_authenticity(self, features):
        """Predict if an artwork is authentic"""
        if features is None:
            return None
            
        features = features.reshape(1, -1)
        prediction = self.model.predict(features)[0]
        probability = self.model.predict_proba(features)[0]
        
        return {
            'is_authentic': bool(prediction),
            'authenticity_score': int(probability[1] * 100),
            'confidence': int(max(probability) * 100)
        }
    
    def save_model(self, filepath):
        """Save the trained model"""
        joblib.dump(self.model, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load a trained model"""
        self.model = joblib.load(filepath)
        print(f"Model loaded from {filepath}")

# Training script
if __name__ == "__main__":
    print("🎭 CraftGuard - Training Cultural Art Authenticity Classifier")
    print("=" * 60)
    
    # Initialize classifier
    classifier = CulturalArtClassifier()
    
    # Create sample dataset
    dataset = classifier.create_sample_dataset()
    print(f"Dataset created with {len(dataset)} samples")
    
    # Train the model
    accuracy = classifier.train(dataset)
    
    # Save the model
    os.makedirs('models', exist_ok=True)
    classifier.save_model('models/cultural_art_classifier.pkl')
    
    # Test with sample data
    print("\n" + "=" * 60)
    print("Testing with sample features...")
    
    # Test authentic sample
    authentic_features = np.array([150, 0.3, 20, 1000, 0.7, 80, 120, 0.15, 1.2])
    result = classifier.predict_authenticity(authentic_features)
    print(f"Authentic sample prediction: {result}")
    
    # Test fake sample
    fake_features = np.array([80, 0.8, 40, 500, 0.9, 30, 180, 0.25, 1.0])
    result = classifier.predict_authenticity(fake_features)
    print(f"Fake sample prediction: {result}")
    
    print("\n🎉 Training completed successfully!")
