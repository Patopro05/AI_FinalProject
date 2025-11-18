# Artificial Intelligence Final Project
"""
This is the main code for the training of a CNN based on X-ray chest scan (with pneumonia or healthy).
The work is purely academic and for learning purposes. It does not intend to replace any medical diagnosis.
This project is Open Source and can be found at the following repository:
Link: https://github.com/Patopro05/AI_FinalProject/edit/main
Team members:
2010356 Fernando Patricio Gutiérrez González
2132103 Axel Muñoz Baca
2050354 Ulises Castillo Díaz
2055281 Pablo Daniel Contreras Obregón
"""
# Importing libraries, such as tensorflow, keras from tensorflow
# and its layers, models and callbacks, matplotlib and os for
# directory management
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import os
# Dataset directory
BASE_DIR = 'Data/chest_xray'
TRAIN_DIR = os.path.join(BASE_DIR, 'train')
TEST_DIR = os.path.join(BASE_DIR, 'test')
# Parameters for dataset loading
IMG_SIZE = (150, 150)
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.2
# Train dataset with binary labels (normal and pneumonia),
# using batches of 32 images, and resized 150x150
train_dataset = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='binary',
    validation_split = VALIDATION_SPLIT,
    subset = 'training',
    seed = 123 )
# Same for validation dataset
val_dataset = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='binary',
    validation_split = VALIDATION_SPLIT,
    subset = 'validation',
    seed = 123 )
# Same for test dataset
test_dataset = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='binary' )
# Names for the classes
class_names = train_dataset.class_names
# Data augmentation
data_augmentation = models.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
    ], name = "data_augmentation" )
# Optimizing
AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.cache().prefetch(buffer_size=AUTOTUNE)
val_dataset = val_dataset.cache().prefetch(buffer_size=AUTOTUNE)
test_dataset = test_dataset.cache().prefetch(buffer_size=AUTOTUNE)
# The model
model = models.Sequential()
model.add(layers.Input(shape = (IMG_SIZE[0], IMG_SIZE[1], 3) ))
model.add(data_augmentation)
model.add(layers.Rescaling(1./255))

model.add(layers.Conv2D(32,(3, 3), activation = 'relu'))
model.add(layers.MaxPooling2D())
model.add(layers.Conv2D(64, (3, 3), activation = 'relu'))
model.add(layers.MaxPooling2D())
model.add(layers.Conv2D(128, (3, 3), activation = 'relu'))
model.add(layers.MaxPooling2D())

model.add(layers.Flatten())
model.add(layers.Dense(128, activation = 'relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation = 'sigmoid'))
# Compiling the model
model.compile(
        optimizer = 'adam',
        loss = 'binary_crossentropy',
        metrics = ['accuracy'] )
# Weight calculation and classification
total_train_files = 4173
weight_for_0 = (1 / 1073) * (total_train_files / 2.0)
weight_for_1 = (1 / 3100) * (total_train_files / 2.0)
class_weights = {0: weight_for_0, 1: weight_for_1}
# Callbacks
early_stopping_cb = EarlyStopping(
        monitor = 'val_loss',
        patience = 5, 
        restore_best_weights = True )
# Train
history = model.fit(
        train_dataset,
        epochs = 20,
        validation_data = val_dataset,
        class_weight = class_weights,
        callbacks = [early_stopping_cb] )
# Model accuracy for both validation and testing
test_loss, test_acc = model.evaluate(test_dataset)
# Final number of epochs
epochs_ran = len(history.history['loss'])
# Graphing plot for Training and Validation
plt.figure(figsize= (12, 5))
plt.subplot(1, 2, 1)
plt.plot(range(epochs_ran), history.history['accuracy'], label = 'Training precission')
plt.plot(range(epochs_ran), history.history['val_accuracy'], label = 'Validation precission')
plt.legend(loc = 'lower right')
plt.title('Precission for Training and Validation')
plt.show()
# Saving the model
model.save('pneumonia_model.keras')