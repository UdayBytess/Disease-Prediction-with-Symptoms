# 🏥 AI-Powered Disease Prediction System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.13-blue?style=for-the-badge&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=for-the-badge&logo=tensorflow)
![Flask](https://img.shields.io/badge/Flask-3.1.0-green?style=for-the-badge&logo=flask)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.6.1-yellow?style=for-the-badge&logo=scikitlearn)

**An intelligent deep learning system that predicts diseases from symptom descriptions with 89.33% accuracy**

[Features](#-features) • [Architecture](#-architecture) • [Installation](#-installation) • [Usage](#-usage) • [Model Performance](#-model-performance)

</div>

---

## 📋 Overview

This project implements a state-of-the-art disease prediction system using advanced Natural Language Processing (NLP) and Deep Learning techniques. By analyzing patient-reported symptoms in natural language, the system accurately identifies potential diseases, serving as an intelligent diagnostic support tool for healthcare applications.

## ✨ Features

### 🧠 Advanced Machine Learning
- **Bidirectional LSTM Networks**: Captures contextual information from both directions in symptom sequences
- **Custom Attention Mechanism**: Focuses on the most relevant symptoms for accurate predictions
- **Deep Neural Architecture**: Multi-layer network with 128-dimensional embeddings and dense layers

### 🎯 High Performance
- **89.33% Accuracy** on validation data
- **Real-time Predictions** with confidence scores
- **Robust Text Processing** handles various input formats
- **Multi-class Classification** across diverse disease categories

### 🌐 Production-Ready Web Interface
- **Flask-powered REST API** for easy integration
- **Interactive Web UI** for symptom input
- **Confidence Scoring** for prediction reliability
- **Responsive Design** for desktop and mobile devices

### 🔧 Technical Excellence
- **Custom Keras Layers** with attention mechanism implementation
- **Comprehensive Preprocessing** including tokenization and text cleaning
- **Model Persistence** with pickle serialization
- **Stratified Data Splitting** for balanced training

## 🏗️ Architecture

### Model Components

```
Input Layer (150 tokens)
    ↓
Embedding Layer (10,000 vocab, 128 dimensions)
    ↓
Bidirectional LSTM (128 units, 30% dropout)
    ↓
Custom Attention Layer
    ↓
Dense Layer (128 units, ReLU activation)
    ↓
Dropout Layer (40%)
    ↓
Output Layer (Softmax, multi-class)
```

### Key Technologies

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Deep Learning** | TensorFlow/Keras | Neural network implementation |
| **NLP** | Tokenization, Padding | Text preprocessing |
| **Web Framework** | Flask 3.1.0 | API and web interface |
| **Data Processing** | Pandas, NumPy | Data manipulation |
| **ML Tools** | scikit-learn | Encoding, metrics, splitting |

## 🚀 Installation

### Prerequisites
- Python 3.13 or higher
- pip package manager

### Setup Instructions

1. **Clone the repository**
```bash
git clone https://github.com/UdayBytess/Disease-Prediction-with-Symptoms
cd disease-prediction-system
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
pip install tensorflow==2.15.0  # Install TensorFlow separately
```

4. **Verify model files**
Ensure these files exist in the project directory:
- `disease_prediction_model.h5`
- `preprocessing.pkl`
- `Symptom2Disease.csv`

## 💻 Usage

### Running the Web Application

```bash
python app.py
```

Visit `http://127.0.0.1:5000` in your browser

### API Endpoint

**POST** `/predict`

**Request:**
```json
{
  "symptoms": "fever headache body pain"
}
```

**Response:**
```json
{
  "prediction": "Dengue",
  "confidence": 92.45,
  "symptoms": "fever headache body pain"
}
```

### Python Integration

```python
from app import predict_disease

symptoms = "persistent cough, chest pain, difficulty breathing"
disease, confidence = predict_disease(symptoms)
print(f"Predicted Disease: {disease} (Confidence: {confidence:.2f}%)")
```

## 📊 Model Performance

### Training Metrics (Final Epoch)

| Metric | Training | Validation |
|--------|----------|------------|
| **Accuracy** | 99.01% | 89.33% |
| **Precision** | 90.93% | - |
| **Recall** | 89.33% | - |
| **F1-Score** | 89.01% | - |
| **Loss** | 0.0399 | 0.3797 |

### Training Configuration
- **Epochs**: 25
- **Batch Size**: 32
- **Optimizer**: Adam
- **Loss Function**: Sparse Categorical Crossentropy
- **Validation Split**: 20%

### Key Performance Highlights
- Rapid convergence (significant improvement by Epoch 5)
- Excellent generalization with controlled overfitting
- Consistent performance across multiple disease categories
- Real-time inference capability

## 🔬 Technical Deep Dive

### Attention Mechanism

The custom attention layer computes weighted importance of symptom tokens:

```python
score = tanh(W·x + b)
attention_weights = softmax(u·score)
context_vector = Σ(attention_weights × hidden_states)
```

### Text Preprocessing Pipeline

1. **Lowercasing**: Standardize text format
2. **Punctuation Removal**: Clean special characters
3. **Whitespace Normalization**: Remove extra spaces
4. **Tokenization**: Convert to numerical sequences
5. **Padding**: Fixed-length input (150 tokens)

### Model Training Features

- **Stratified Splitting**: Maintains disease distribution
- **Dropout Regularization**: Prevents overfitting (30% + 40%)
- **Bidirectional Processing**: Context from both directions
- **Custom Callbacks**: Real-time metric monitoring

## 📁 Project Structure

```
disease-prediction-system/
│
├── app.py                          # Flask application
├── disease_prediction_model.ipynb  # Model training notebook
├── requirements.txt                # Python dependencies
│
├── disease_prediction_model.h5     # Trained model weights
├── preprocessing.pkl               # Tokenizer & encoder
├── Symptom2Disease.csv            # Training dataset
│
└── templates/
    └── index.html                 # Web interface
```

## 🎓 Use Cases

- **Clinical Decision Support**: Assist healthcare providers with preliminary diagnoses
- **Telemedicine Platforms**: Enable remote symptom assessment
- **Health Chatbots**: Integrate with conversational AI systems
- **Medical Education**: Training tool for medical students
- **Health Monitoring Apps**: Personal health assessment features

## 🔮 Future Enhancements

- [ ] Multi-language support for global accessibility
- [ ] Explainability features (SHAP, LIME) for interpretable predictions
- [ ] Integration with medical knowledge graphs
- [ ] Mobile application development
- [ ] Continuous learning from new medical data
- [ ] Risk assessment and severity scoring
- [ ] Medical history context integration

## 🛡️ Disclaimer

This system is designed as a **diagnostic support tool** and should not replace professional medical advice. Always consult qualified healthcare providers for medical decisions.

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

For questions or collaborations, please open an issue or contact the maintainers.

---

<div align="center">

**Made with ❤️ using TensorFlow and Flask**

⭐ Star this repository if you find it helpful!

</div>
