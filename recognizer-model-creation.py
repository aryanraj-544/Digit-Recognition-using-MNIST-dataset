import numpy as np
import matplotlib.pyplot as plt
import os
import keras 
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Get Data and Pre-Process the Data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(f"Training data shape: {x_train.shape}, {y_train.shape}")
print(f"Test data shape: {x_test.shape}, {y_test.shape}")

def plot_input_img(i):
    plt.imshow(x_train[i], cmap='binary')
    plt.title(f"Label: {y_train[i]}")
    plt.show()

# Display sample images
for i in range(10):
    plot_input_img(i)

# Pre Process the images
x_train = x_train.astype(np.float32) / 255.0
x_test = x_test.astype(np.float32) / 255.0

# Reshape / expand dimensions of images to (28,28,1)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

print(f"After preprocessing - Training shape: {x_train.shape}")
print(f"After preprocessing - Test shape: {x_test.shape}")

# One hot encode the labels
y_train_categorical = keras.utils.to_categorical(y_train, 10)
y_test_categorical = keras.utils.to_categorical(y_test, 10)

print(f"Original labels shape: {y_train.shape}")
print(f"One-hot encoded labels shape: {y_train_categorical.shape}")

# Build improved model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    BatchNormalization(),
    Conv2D(32, (3,3), activation='relu'),
    MaxPool2D((2,2)),
    Dropout(0.25),
    
    Conv2D(64, (3,3), activation='relu'),
    BatchNormalization(),
    Conv2D(64, (3,3), activation='relu'),
    MaxPool2D((2,2)),
    Dropout(0.25),
    
    Conv2D(128, (3,3), activation='relu'),
    BatchNormalization(),
    Dropout(0.25),
    
    Flatten(),
    Dense(512, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(10, activation='softmax')
])

model.summary()

# Compile model with better optimizer settings
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Improved callbacks
es = EarlyStopping(
    monitor='val_accuracy',  # Updated metric name
    min_delta=0.001,
    patience=7,
    verbose=1,
    restore_best_weights=True
)

mc = ModelCheckpoint(
    './recognizer_model.keras',
    monitor='val_accuracy',  # Updated metric name
    verbose=1,
    save_best_only=True,
    save_weights_only=False
)

# Learning rate reduction callback
lr_reduce = ReduceLROnPlateau(
    monitor='val_accuracy',
    factor=0.2,
    patience=3,
    min_lr=0.0001,
    verbose=1
)

callbacks = [es, mc, lr_reduce]

# Data augmentation (optional but recommended)
from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    fill_mode='nearest'
)

# Model Training with data augmentation
print("Starting model training...")
history = model.fit(
    datagen.flow(x_train, y_train_categorical, batch_size=128),
    steps_per_epoch=len(x_train) // 128,
    epochs=30,
    validation_data=(x_test, y_test_categorical),
    callbacks=callbacks,
    verbose=1
)

# Alternative: Training without data augmentation (uncomment if you prefer)
# history = model.fit(
#     x_train, y_train_categorical,
#     batch_size=128,
#     epochs=30,
#     validation_split=0.2,  # Use 20% for validation
#     callbacks=callbacks,
#     verbose=1
# )

# Load the best model
model_best = keras.models.load_model("recognizer_model.keras")

# Evaluate the model
test_loss, test_accuracy = model_best.evaluate(x_test, y_test_categorical, verbose=0)
print(f'\nTest Loss: {test_loss:.4f}')
print(f'Test Accuracy: {test_accuracy:.4f}')

# Make predictions
y_pred = model_best.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred_classes))

# Confusion matrix
plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_test, y_pred_classes)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

# Plot training history
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

# Test with some sample predictions
def test_predictions(num_samples=10):
    indices = np.random.choice(len(x_test), num_samples, replace=False)
    
    plt.figure(figsize=(15, 6))
    for i, idx in enumerate(indices):
        plt.subplot(2, 5, i+1)
        plt.imshow(x_test[idx].reshape(28, 28), cmap='binary')
        
        pred = model_best.predict(x_test[idx].reshape(1, 28, 28, 1))
        pred_label = np.argmax(pred)
        confidence = np.max(pred) * 100
        
        plt.title(f'True: {y_test[idx]}, Pred: {pred_label}\nConf: {confidence:.1f}%')
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

test_predictions()

print(f"\nModel saved successfully at: recognizer_model.keras")
print(f"Final test accuracy: {test_accuracy:.4f}")