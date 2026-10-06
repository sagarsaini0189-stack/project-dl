# STEP 1: Import core Deep Learning and Image Processing Libraries
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# STEP 2: Simulate a structured Medical Image Dataset (Clean Matrix Arrays)
# We generate a tiny batch of 6 mock X-ray scans (each scan is a 32x32 pixel grayscale grid)
np.random.seed(42)
X_train = np.random.rand(6, 32, 32, 1)  # 6 images, 32x32 size, 1 color channel (grayscale)

# Labels: 0 = Normal / Healthy Scan, 1 = Anomaly / Tumor Detected
y_train = np.array([0, 1, 0, 1, 0, 1])

print("--- Medical Imaging Diagnostic Dataset ---")
print(f"Total Structural Scans Loaded: {X_train.shape[0]}")
print(f"Image Resolution Array: {X_train.shape[1]}x{X_train.shape[2]} pixels")
print("------------------------------------------")

# STEP 3: Build the Convolutional Neural Network (CNN) Architecture
# CNNs are specialized neural nets designed to extract geometric shapes from visual data
model = Sequential([
    # Convolutional Layer: Detects sharp edges, shadows, and internal patterns in scans
    Conv2D(8, (3, 3), activation='relu', input_shape=(32, 32, 1)),

    # Pooling Layer: Downsamples the dimensions to make calculations faster
    MaxPooling2D((2, 2)),s

    # Flatten Layer: Converts the 2D visual layout into a 1D line of numbers
    Flatten(),

    # Dense Hidden Layer: Connects features to map biological structures
    Dense(16, activation='relu'),

    # Output Layer: Sigmoid forces the final prediction to be a probability between 0 and 1
    Dense(1, activation='sigmoid')
])

# STEP 4: Compile the Diagnostic Neural Engine
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# STEP 5: Train the Deep Medical Brain
# Epochs = 15 means the network reviews the clinical scans 15 times to learn textures
print("\n--- Training the Medical Diagnostic Neural Network ---")
model.fit(X_train, y_train, epochs=15, verbose=1)
print("[Success] Diagnostic engine trained successfully on image data!\n")

# STEP 6: Practical Diagnostic Testing (Inference Evaluation)
# Let's feed the model a brand new, unseen incoming patient X-ray/MRI matrix
new_patient_scan = np.random.rand(1, 32, 32, 1)

# Execute the diagnostic evaluation
prediction = model.predict(new_patient_scan, verbose=0)

print("--- Clinical Inference Evaluation ---")
print("Incoming Scan Analysis Complete.")
if prediction > 0.5:
    print(f"System Alert: ANOMALY / TUMOR DETECTED! (Confidence Score: {prediction[0][0]:.2f})")
    print("Action Required: Forwarding to Senior Radiologist.")
else:
    print(f"System Status: SCAN NORMAL / HEALTHY. (Confidence Score: {prediction[0][0]:.2f})")