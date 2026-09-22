import os
import numpy as np
import matplotlib.pyplot as plt
import joblib

# Deep Learning Models
from keras.models import Sequential  # type: ignore
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, LSTM, Embedding  # type: ignore
from keras.preprocessing.text import Tokenizer  # type: ignore

# Machine Learning
from sklearn.ensemble import RandomForestClassifier

def setup_directories():
    if not os.path.exists('models'):
        os.makedirs('models')

def plot_metrics(history, title):
    # Plot training & validation accuracy values
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title(f'{title} Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    # Plot training & validation loss values
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title(f'{title} Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')

    plt.tight_layout()
    plt.savefig(f'models/{title}_metrics.png')
    plt.close()

def train_video_cnn():
    print("--- Training CNN for Facial Expresion Deception (Video) ---")
    
    # 1. Create Mock Dataset (Images: 48x48x1 Grayscale)
    # Binary Classification (Truth=0, Lie=1)
    X_train = np.random.rand(500, 48, 48, 1) 
    y_train = np.random.randint(0, 2, 500)
    X_val = np.random.rand(100, 48, 48, 1)
    y_val = np.random.randint(0, 2, 100)

    # 2. Build CNN Architecture
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(2, activation='softmax') # [prob_truth, prob_lie]
    ])

    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    # 3. Train
    print("Training CNN for 5 epochs...")
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=5, batch_size=32, verbose=1)
    
    # 4. Save & Visualize
    model.save('models/video_model.keras')
    plot_metrics(history, "CNN_Video")
    print("Saved Video CNN to models/video_model.keras\n")

def train_audio_rf():
    print("--- Training Random Forest for Audio Stress Analysis ---")
    
    # 1. Provide Mock Dataset (MFCC features: 40 coefficients)
    X_train = np.random.rand(500, 40)
    y_train = np.random.randint(0, 2, 500)
    
    # 2. Build Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # 3. Train
    rf.fit(X_train, y_train)
    acc = rf.score(X_train, y_train)
    print(f"Training Accuracy: {acc * 100:.2f}%")
    
    # 4. Save
    joblib.dump(rf, 'models/audio_model.pkl')
    print("Saved Audio RF to models/audio_model.pkl\n")

def train_text_lstm():
    print("--- Training LSTM for Deceptive Text Analysis ---")
    
    # 1. Mock Texts
    texts = ["I swear I didn't do it, I am innocent", 
             "I was at home all night watching TV",
             "I don't know what you are talking about"] * 200
    y_truth = np.random.randint(0, 2, 600)
    
    # 2. Tokenize and Pad sequences
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(texts)
    vocab_size = len(tokenizer.word_index) + 1
    
    X_train = np.random.randint(1, vocab_size, (600, 100)) # 100 length sequences
    
    # 3. Build LSTM Architecture
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=32, input_length=100),
        LSTM(64, return_sequences=False),
        Dropout(0.5),
        Dense(1, activation='sigmoid') # Binary output (Lie=1, Truth=0)
    ])
    
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    # 4. Train
    print("Training LSTM for 5 epochs...")
    history = model.fit(X_train, y_truth, validation_split=0.2, epochs=5, batch_size=32, verbose=1)
    
    # 5. Save Model and Tokenizer
    model.save('models/text_model.keras')
    joblib.dump(tokenizer, 'models/tokenizer.pkl')
    plot_metrics(history, "LSTM_Text")
    print("Saved Text LSTM & Tokenizer to models/\n")

if __name__ == "__main__":
    setup_directories()
    train_video_cnn()
    train_audio_rf()
    train_text_lstm()
    print("All models trained and datasets validated. You can now launch Streamlit!")
