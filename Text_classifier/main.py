import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Step 1: Preprocess New Data
def preprocess_new_data(new_texts, maxlen, tokenizer):
    new_sequences = tokenizer.texts_to_sequences(new_texts)
    new_padded_sequences = pad_sequences(new_sequences, maxlen=maxlen)
    return new_padded_sequences

# Step 2: Load Trained Model
def load_model(file_path):
    model = tf.keras.models.load_model(file_path)
    return model

# Step 3: Make Predictions
def predict_labels(model, new_padded_sequences):
    predictions = model.predict(new_padded_sequences)
    return predictions

# Example data
new_texts = ['This is a neutral sentence.', 'Another positive example.']

# Load tokenizer and maxlen
tokenizer = Tokenizer()
tokenizer.fit_on_texts(new_texts)
maxlen = max(len(seq) for seq in tokenizer.texts_to_sequences(new_texts))

# Load trained model
loaded_model = load_model('trained_model.h5')

# Preprocess new data
new_padded_sequences = preprocess_new_data(new_texts, maxlen, tokenizer)

# Make predictions
predictions = predict_labels(loaded_model, new_padded_sequences)
print(predictions)
