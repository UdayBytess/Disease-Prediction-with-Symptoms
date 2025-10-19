from flask import Flask, render_template, request
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
import tensorflow as tf

# Initialize Flask app
app = Flask(__name__)

# Define custom AttentionLayer
class AttentionLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name='attention_weight', shape=(input_shape[-1], input_shape[-1]),
                                 initializer='glorot_uniform', trainable=True)
        self.b = self.add_weight(name='attention_bias', shape=(input_shape[-1],),
                                 initializer='zeros', trainable=True)
        self.u = self.add_weight(name='context_vector', shape=(input_shape[-1],),
                                 initializer='glorot_uniform', trainable=True)
        super(AttentionLayer, self).build(input_shape)

    def call(self, x):
        score = tf.nn.tanh(tf.tensordot(x, self.W, axes=[2, 0]) + self.b)
        attention_weights = tf.nn.softmax(tf.tensordot(score, self.u, axes=[2, 0]), axis=1)
        context_vector = tf.reduce_sum(attention_weights[..., tf.newaxis] * x, axis=1)
        return context_vector

    def get_config(self):
        config = super(AttentionLayer, self).get_config()
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

# Load the trained model with custom objects
model = load_model("disease_prediction_model.h5", custom_objects={'AttentionLayer': AttentionLayer})

# Load preprocessing objects (tokenizer, label_encoder)
with open("preprocessing.pkl", "rb") as f:
    preprocessing = pickle.load(f)

tokenizer = preprocessing["tokenizer"]
label_encoder = preprocessing["label_encoder"]

# Define text cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove punctuation
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text

# Function to predict disease
def predict_disease(symptoms):
    symptoms_cleaned = clean_text(symptoms)
    seq = tokenizer.texts_to_sequences([symptoms_cleaned])
    pad = pad_sequences(seq, maxlen=150)
    pred = model.predict(pad)[0]
    predicted_index = np.argmax(pred)
    predicted_label = label_encoder.inverse_transform([predicted_index])[0]
    confidence = pred[predicted_index] * 100  # Convert to percentage
    return predicted_label, confidence

# Define routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    symptoms = request.form['symptoms']
    prediction, confidence = predict_disease(symptoms)
    return render_template('index.html', prediction=prediction, confidence=confidence, symptoms=symptoms)

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
