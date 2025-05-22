# %%
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models


# %%
base_dir = r"D:\chrome download\DL_LAB_EXAM\DL_LAB_EXAM\Datasets\Plant_data\Potato"

# %%
IMG_SIZE = (128, 128)
BATCH_SIZE = 32

# %%
datagen = ImageDataGenerator(rescale=1./255)

# %%
train_generator = datagen.flow_from_directory(
    os.path.join(base_dir, 'train'),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)


# %%
val_generator = datagen.flow_from_directory(
    os.path.join(base_dir, 'val'),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# %%
test_generator = datagen.flow_from_directory(
    os.path.join(base_dir, 'test'),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# %%
def show_sample_images(generator):
    images, labels = next(generator)
    class_names = list(generator.class_indices.keys())

    plt.figure(figsize=(12, 4))
    for i in range(5):
        plt.subplot(1, 5, i+1)
        plt.imshow(images[i])
        plt.title(class_names[labels[i].argmax()])
        plt.axis('off')
    plt.tight_layout()
    plt.show()

show_sample_images(train_generator)

# %%
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(train_generator.num_classes, activation='softmax')
])

# %%
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# %%
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=5
)

# %%
# 📈 Plot Training and Validation Accuracy/Loss
plt.figure(figsize=(12, 5))

# Accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training & Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training & Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

# %%
test_loss, test_accuracy = model.evaluate(test_generator)
print(f"\n✅ Test Accuracy: {test_accuracy:.4f}")
print(f"❌ Test Loss: {test_loss:.4f}")

# %%
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Get true labels and predictions
y_true = test_generator.classes
y_pred_probs = model.predict(test_generator)
y_pred = np.argmax(y_pred_probs, axis=1)

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
class_names = list(test_generator.class_indices.keys())

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()


# %%
# Evaluate on train set
train_loss, train_accuracy = model.evaluate(train_generator)
print(f"\n📘 Final Training Accuracy: {train_accuracy:.4f}")
print(f"📘 Training Loss: {train_loss:.4f}")

# Evaluate on validation set
val_loss, val_accuracy = model.evaluate(val_generator)
print(f"\n📗 Validation Accuracy: {val_accuracy:.4f}")
print(f"📗 Validation Loss: {val_loss:.4f}")

# Evaluate on test set
test_loss, test_accuracy = model.evaluate(test_generator)
print(f"\n📕 Test Accuracy: {test_accuracy:.4f}")
print(f"📕 Test Loss: {test_loss:.4f}")



