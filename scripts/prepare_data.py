import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import json

def create_cultural_art_metadata():
    """Create metadata for different cultural art forms"""
    
    cultural_arts = {
        "madhubani": {
            "name": "Madhubani",
            "origin": "Bihar, India",
            "characteristics": {
                "patterns": ["geometric", "floral", "animal_motifs", "religious_symbols"],
                "colors": ["red", "yellow", "blue", "green", "black"],
                "techniques": ["fine_lines", "natural_dyes", "hand_painted"],
                "materials": ["handmade_paper", "cloth", "natural_pigments"]
            },
            "authenticity_markers": {
                "color_variation": "high",
                "pattern_precision": "medium",
                "symmetry": "low_to_medium",
                "brush_strokes": "visible_variations"
            }
        },
        "warli": {
            "name": "Warli",
            "origin": "Maharashtra, India",
            "characteristics": {
                "patterns": ["geometric_shapes", "stick_figures", "tribal_motifs", "nature_scenes"],
                "colors": ["white", "red", "brown"],
                "techniques": ["simple_geometric", "white_pigment", "tribal_style"],
                "materials": ["mud_walls", "natural_white_pigment", "bamboo_brushes"]
            },
            "authenticity_markers": {
                "color_variation": "low",
                "pattern_precision": "low",
                "symmetry": "low",
                "brush_strokes": "rough_natural"
            }
        },
        "kalamkari": {
            "name": "Kalamkari",
            "origin": "Andhra Pradesh, India",
            "characteristics": {
                "patterns": ["floral", "mythological", "nature_inspired", "intricate_details"],
                "colors": ["indigo", "red", "black", "yellow", "green"],
                "techniques": ["hand_painted", "block_printed", "natural_dyes"],
                "materials": ["cotton_fabric", "natural_dyes", "bamboo_pen"]
            },
            "authenticity_markers": {
                "color_variation": "high",
                "pattern_precision": "high",
                "symmetry": "medium",
                "brush_strokes": "fine_detailed"
            }
        },
        "banarasi": {
            "name": "Banarasi",
            "origin": "Varanasi, India",
            "characteristics": {
                "patterns": ["brocade", "floral", "geometric", "paisley"],
                "colors": ["gold", "silver", "red", "green", "purple"],
                "techniques": ["silk_weaving", "gold_thread", "intricate_weaving"],
                "materials": ["silk_fabric", "gold_thread", "silver_thread"]
            },
            "authenticity_markers": {
                "color_variation": "medium",
                "pattern_precision": "very_high",
                "symmetry": "high",
                "brush_strokes": "woven_texture"
            }
        }
    }
    
    return cultural_arts

def generate_training_samples():
    """Generate synthetic training data based on cultural art characteristics"""
    
    print("Generating training samples for cultural art detection...")
    
    samples = []
    labels = []
    art_forms = []
    
    # Get cultural art metadata
    cultural_arts = create_cultural_art_metadata()
    
    # Generate authentic samples for each art form
    for art_form, metadata in cultural_arts.items():
        print(f"Generating authentic samples for {metadata['name']}...")
        
        for i in range(50):  # 50 authentic samples per art form
            # Generate features based on authenticity markers
            markers = metadata['authenticity_markers']
            
            # Color variation
            if markers['color_variation'] == 'high':
                color_var = np.random.normal(150, 40)
            elif markers['color_variation'] == 'medium':
                color_var = np.random.normal(100, 30)
            else:
                color_var = np.random.normal(70, 20)
            
            # Pattern precision (lower for authentic handmade)
            if markers['pattern_precision'] == 'very_high':
                pattern_reg = np.random.normal(0.7, 0.1)
            elif markers['pattern_precision'] == 'high':
                pattern_reg = np.random.normal(0.5, 0.1)
            elif markers['pattern_precision'] == 'medium':
                pattern_reg = np.random.normal(0.3, 0.1)
            else:
                pattern_reg = np.random.normal(0.2, 0.1)
            
            # Other features
            edge_sharp = np.random.normal(25, 10)
            texture_comp = np.random.normal(1200, 300)
            symmetry = np.random.normal(0.6, 0.2)
            brush_var = np.random.normal(90, 25)
            color_sat = np.random.normal(130, 35)
            detail_dens = np.random.normal(0.18, 0.05)
            geom_ratio = np.random.normal(1.3, 0.4)
            
            sample = [color_var, pattern_reg, edge_sharp, texture_comp, 
                     symmetry, brush_var, color_sat, detail_dens, geom_ratio]
            
            samples.append(sample)
            labels.append(1)  # 1 = authentic
            art_forms.append(art_form)
    
    # Generate fake/machine-made samples
    print("Generating fake/machine-made samples...")
    
    for i in range(100):  # 100 fake samples
        # Fake samples have different characteristics
        color_var = np.random.normal(60, 15)      # Lower color variation
        pattern_reg = np.random.normal(0.85, 0.1) # Higher pattern regularity
        edge_sharp = np.random.normal(45, 15)     # Sharper edges
        texture_comp = np.random.normal(400, 100) # Lower texture complexity
        symmetry = np.random.normal(0.95, 0.05)   # Higher symmetry
        brush_var = np.random.normal(25, 8)       # Lower brush variation
        color_sat = np.random.normal(200, 30)     # Higher saturation
        detail_dens = np.random.normal(0.3, 0.1)  # Higher detail density
        geom_ratio = np.random.normal(1.0, 0.1)   # More regular ratios
        
        sample = [color_var, pattern_reg, edge_sharp, texture_comp, 
                 symmetry, brush_var, color_sat, detail_dens, geom_ratio]
        
        samples.append(sample)
        labels.append(0)  # 0 = fake
        art_forms.append('fake')
    
    # Create DataFrame
    feature_names = [
        'color_variance', 'pattern_regularity', 'edge_sharpness', 
        'texture_complexity', 'symmetry_score', 'brush_stroke_variation',
        'color_saturation', 'detail_density', 'geometric_ratio'
    ]
    
    df = pd.DataFrame(samples, columns=feature_names)
    df['authentic'] = labels
    df['art_form'] = art_forms
    
    return df, cultural_arts

def save_datasets(df, cultural_arts):
    """Save datasets and metadata"""
    
    # Create data directory
    os.makedirs('data', exist_ok=True)
    
    # Save training data
    df.to_csv('data/training_data.csv', index=False)
    print(f"Training data saved: {len(df)} samples")
    
    # Save cultural arts metadata
    with open('data/cultural_arts_metadata.json', 'w') as f:
        json.dump(cultural_arts, f, indent=2)
    print("Cultural arts metadata saved")
    
    # Create train/test split
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['authentic'])
    
    train_df.to_csv('data/train_data.csv', index=False)
    test_df.to_csv('data/test_data.csv', index=False)
    
    print(f"Train set: {len(train_df)} samples")
    print(f"Test set: {len(test_df)} samples")
    
    # Print statistics
    print("\nDataset Statistics:")
    print("Authentic vs Fake distribution:")
    print(df['authentic'].value_counts())
    print("\nArt form distribution:")
    print(df['art_form'].value_counts())

if __name__ == "__main__":
    print("🧢 CraftGuard - Data Preparation")
    print("=" * 50)
    
    # Generate training data
    df, cultural_arts = generate_training_samples()
    
    # Save datasets
    save_datasets(df, cultural_arts)
    
    print("\n✅ Data preparation completed successfully!")
    print("Next steps:")
    print("1. Run train_model.py to train the classifier")
    print("2. Run api_server.py to start the API server")
    print("3. Open index.html to use the web interface")
