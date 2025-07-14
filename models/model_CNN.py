import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Dense, Dropout, Activation, Flatten,
                                     Conv2D, MaxPooling2D, BatchNormalization)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical

# Input image dimensions
img_rows, img_cols = 64, 64
img_channels = 3  # RGB Images
batch_size = 32
num_classes = 7
epochs = 100
learning_rate = 1e-4

# Path for images
data_path = 'C:\\Users\\shital\\Desktop\\python\\dataset\\training_set'
processed_path = 'C:\\Users\\shital\\Desktop\\python\\training_set'

# Ensure processed path exists
os.makedirs(processed_path, exist_ok=True)

# Load and process images
images = []
labels = []
for label, folder in enumerate(sorted(os.listdir(data_path))):
    folder_path = os.path.join(data_path, folder)

    # ✅ Only proceed if it's a directory (to skip .h5 or other files)
    if not os.path.isdir(folder_path):
        continue

    for file in os.listdir(folder_path):
        img_path = os.path.join(folder_path, file)
        try:
            img = Image.open(img_path).resize((img_rows, img_cols)).convert('RGB')
            images.append(np.array(img))
            labels.append(label)
        except Exception as e:
            print(f"Skipping file {img_path} due to error: {e}")

images = np.array(images, dtype='float32') / 255.0  # Normalize to [0, 1]
labels = np.array(labels)

# Check unique labels and their range
print(f"Unique labels: {np.unique(labels)}")

# Ensure labels are within the valid range [0, num_classes - 1]
if np.max(labels) >= num_classes or np.min(labels) < 0:
    print("Labels are out of range! Correcting...")
    labels = np.clip(labels, 0, num_classes - 1)

# Proceed with train-test split
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Convert labels to categorical
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

print(f"Training samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")

# Model architecture
model = Sequential()
model.add(Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(img_rows, img_cols, img_channels)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Dropout(0.25))

model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Dropout(0.25))

model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(optimizer=Adam(learning_rate=learning_rate),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# Callbacks
callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ModelCheckpoint('best_model.h5', save_best_only=True, monitor='val_loss', mode='min')
]

# Training the model
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    batch_size=batch_size,
    epochs=epochs,
    callbacks=callbacks,
    verbose=1
)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test loss: {test_loss:.4f}, Test accuracy: {test_accuracy:.4f}")

# Classification metrics
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

print("Classification Report:")
print(classification_report(y_true_classes, y_pred_classes))
print("Confusion Matrix:")
print(confusion_matrix(y_true_classes, y_pred_classes))

# Save the model
model.save('final_model.h5')

# Plot training history
plt.figure(figsize=(12, 4))

# Loss Plot
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Accuracy Plot
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()
