# 🧢 CraftGuard - AI Cultural Art Authenticity Detector

**Protecting Cultural Heritage Through AI Technology**

CraftGuard is an AI-powered web application that helps detect fake cultural products online. It analyzes images of traditional art forms like Madhubani, Warli, Kalamkari, and Banarasi to determine authenticity and protect both customers and artisans.

## 🎯 Problem Statement

- **Customers are deceived** by fake cultural products sold as authentic
- **Traditional artisans lose income** to mass-produced imitations  
- **Cultural heritage is diluted** through commercialization

## 💡 Solution

CraftGuard uses machine learning to:
- ✅ Analyze patterns, colors, and craftsmanship details
- ✅ Detect machine-made vs handmade characteristics
- ✅ Provide authenticity scores with confidence levels
- ✅ Connect users with verified artisans

## 🚀 Features

- **Image Upload & Analysis** - Drag & drop or URL input
- **AI Pattern Detection** - 9 different authenticity markers
- **Cultural Art Database** - Information on 4 major art forms
- **Authenticity Scoring** - Percentage-based results
- **Verified Sellers** - Links to trusted artisan sources
- **Educational Tips** - How to identify authentic cultural art

## 🛠️ Technology Stack

**Frontend:**
- HTML5, CSS3, JavaScript (ES6+)
- Responsive design with modern UI/UX
- Drag & drop file upload

**Backend:**
- Python 3.8+
- Flask API server
- OpenCV for image processing
- Scikit-learn for ML classification
- NumPy, Pandas for data handling

**Machine Learning:**
- Random Forest Classifier
- Computer Vision feature extraction
- Pattern recognition algorithms

## 📦 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Modern web browser

### Setup Instructions

1. **Clone the repository**
\`\`\`bash
git clone https://github.com/yourusername/craftguard.git
cd craftguard
\`\`\`

2. **Install Python dependencies**
\`\`\`bash
pip install -r requirements.txt
\`\`\`

3. **Prepare training data**
\`\`\`bash
python scripts/prepare_data.py
\`\`\`

4. **Train the ML model**
\`\`\`bash
python scripts/train_model.py
\`\`\`

5. **Start the API server**
\`\`\`bash
python scripts/api_server.py
\`\`\`

6. **Open the web application**
\`\`\`bash
# Open index.html in your browser
# Or serve with a local server:
python -m http.server 8080
\`\`\`

## 🎨 Supported Cultural Art Forms

| Art Form | Origin | Key Features |
|----------|--------|--------------|
| **Madhubani** | Bihar, India | Geometric patterns, natural dyes, fine lines |
| **Warli** | Maharashtra, India | Simple geometric shapes, tribal motifs |
| **Kalamkari** | Andhra Pradesh, India | Hand-painted, natural dyes, floral patterns |
| **Banarasi** | Varanasi, India | Silk fabric, brocade work, gold threads |

## 🔬 How It Works

1. **Feature Extraction** - Analyzes 9 key characteristics:
   - Color variance and saturation
   - Pattern regularity and symmetry
   - Edge sharpness and texture complexity
   - Brush stroke variations
   - Detail density and geometric ratios

2. **ML Classification** - Random Forest model trained on:
   - 200+ authentic art samples
   - 100+ machine-made samples
   - Cultural art form characteristics

3. **Authenticity Scoring** - Provides:
   - Percentage authenticity score
   - Confidence level
   - Detailed feature analysis
   - Risk factor identification

## 📊 Model Performance

- **Accuracy**: ~85-90% on test data
- **Precision**: High for detecting machine-made products
- **Recall**: Good for identifying authentic handmade art
- **Features**: 9 computer vision-based authenticity markers

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md).

### Areas for Contribution:
- Additional cultural art forms
- Improved ML models (CNN, deep learning)
- Mobile app development
- Database expansion
- UI/UX improvements

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Traditional artisans who preserve cultural heritage
- Cultural art researchers and historians
- Open source computer vision community
- Contributors and supporters of authentic cultural art

## 📞 Contact

- **Project Maintainer**: [Your Name]
- **Email**: [your.email@example.com]
- **Website**: [https://craftguard.example.com]

## 🔮 Future Roadmap

- [ ] Deep learning models (CNN, ResNet)
- [ ] Mobile app (React Native/Flutter)
- [ ] Blockchain verification system
- [ ] Marketplace integration
- [ ] Real-time scanning API
- [ ] Artisan certification platform

---

**⭐ Star this repository if you find it helpful!**

**🤝 Help us protect cultural heritage through technology**
