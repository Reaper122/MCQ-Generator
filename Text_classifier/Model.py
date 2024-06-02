import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense
import matplotlib.pyplot as plt

# Step 1: Preprocess Data
def preprocess_data(texts, labels):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    maxlen = max(len(seq) for seq in sequences)
    padded_sequences = pad_sequences(sequences, maxlen=maxlen)
    vocab_size = len(tokenizer.word_index) + 1
    return padded_sequences, labels, maxlen, vocab_size, tokenizer

# Step 2: Build Model
def build_model(maxlen, vocab_size):
    embedding_dim = 50
    num_filters = 64
    kernel_size = 5
    
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=maxlen),
        Conv1D(num_filters, kernel_size, activation='relu'),
        GlobalMaxPooling1D(),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

# Step 3: Train Model
def train_model(model, X_train, y_train, X_val, y_val, epochs=5, batch_size=32):
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val))
    return history

# Step 4: Evaluate Model
def evaluate_model(model, X_test, y_test):
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f'Test Loss: {test_loss}, Test Accuracy: {test_accuracy}')

# Step 5: Plot Training History and Save Plot
def plot_and_save_history(history, filename='training_history.png'):
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.savefig(filename)
    plt.close()

# Example data
texts = ['This is a positive sentence.', 'This is a negative sentence.', 'Another positive example.', 'More negative text here.']
labels = np.array([1, 0, 1, 0])  # 1 for positive, 0 for negative

# Step 1: Preprocess Data
X, y, maxlen, vocab_size, tokenizer = preprocess_data(texts, labels)

# Step 2: Split Data into Train and Validation Sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Build Model
model = build_model(maxlen, vocab_size)

# Step 4: Train Model
history = train_model(model, X_train, y_train, X_val, y_val)

# Step 5: Evaluate Model
evaluate_model(model, X_val, y_val)

# Step 6: Plot and Save Training History
plot_and_save_history(history, filename='training_history.png')

# Step 7: Save Trained Model
model.save('trained_model.h5')

# Step 8: Load Trained Model
# loaded_model = tf.keras.models.load_model('trained_model.h5')

# Step 9: Make Predictions on Different Example
# new_texts = ['This is a neutral sentence.', 'Another positive example.']
# new_sequences = tokenizer.texts_to_sequences(new_texts)
# new_padded_sequences = pad_sequences(new_sequences, maxlen=maxlen)
# predictions = loaded_model.predict(new_padded_sequences)
# print(predictions)
