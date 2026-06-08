import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras
from tensorflow.keras import layers
import pickle

# 1. Load dataset
df = pd.read_csv("dataset.csv", header=None)
X = df.iloc[:, :-1].values   # 63 landmark features
y = df.iloc[:, -1].values    # gesture labels (text)

# 2. Convert gesture names to numbers
# Neural networks need numbers not text like "hello", "fist" etc
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)  # "hello"→0, "fist"→1 etc

# 3. Convert to one-hot encoding
# Instead of y=3, network sees y=[0,0,0,1,0,0...]
y_onehot = keras.utils.to_categorical(y_encoded, num_classes=25)

# 4. Train/test split (same as Random Forest for fair comparison)
X_train, X_test, y_train, y_test = train_test_split(
    X, y_onehot, test_size=0.2, random_state=42
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

# 5. Build the Neural Network
model = keras.Sequential([
    layers.Input(shape=(63,)),         # 63 landmark coordinates coming in
    layers.Dense(128, activation='relu'),  # Hidden layer 1 - finds patterns
    layers.Dense(64, activation='relu'),   # Hidden layer 2 - refines patterns
    layers.Dense(25, activation='softmax') # Output - 25 gestures, picks most likely
])

model.summary()  # prints a nice table of your network structure

# 6. Compile - tell the model HOW to learn
model.compile(
    optimizer='adam',        # how it adjusts itself during learning
    loss='categorical_crossentropy',  # how it measures its mistakes
    metrics=['accuracy']     # what we want to see during training
)

# 7. Train the model
history = model.fit(
    X_train, y_train,
    epochs=50,               # how many times it sees the full dataset
    batch_size=32,           # how many samples it learns from at once
    validation_split=0.1,    # keeps 10% of training data to check progress
    verbose=1                # shows progress bar
)

# 8. Evaluate on test set
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"\nNeural Network Accuracy: {test_accuracy * 100:.2f}%")
print(f"Random Forest Accuracy:  96.00%")
print(f"Winner: {'Neural Network' if test_accuracy > 0.96 else 'Random Forest'}")

# 9. Save the model as .h5 file (your Week 4-5 deliverable)
model.save("gesture_model.h5")
print("\nModel saved as gesture_model.h5 ✅")

# 10. Save the encoder too (needed later to convert numbers back to gesture names)
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(encoder, f)
print("Label encoder saved ✅")