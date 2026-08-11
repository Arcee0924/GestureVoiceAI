import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle

os.makedirs("models", exist_ok=True)

# Load dataset safely
data = pd.read_csv("gesture_dataset.csv", header=None, low_memory=False)

# Separate features and labels
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].astype(str).values   # convert labels to string

# Encode labels
encoder = LabelEncoder()
y = encoder.fit_transform(y)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model architecture
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X.shape[1],)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(len(np.unique(y)), activation='softmax')
])

# Compile model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train model
model.fit(X_train, y_train, epochs=30, batch_size=32)

# Evaluate model
loss, acc = model.evaluate(X_test, y_test)
print("Test Accuracy:", acc)

# Save model
model.save("models/gesture_model.h5")

# Save label encoder
pickle.dump(encoder, open("models/label_encoder.pkl", "wb"))

print("Model trained successfully")