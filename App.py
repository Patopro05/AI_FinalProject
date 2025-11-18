import os
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

def load_and_predict():
    # 1. Check if model exists
    model_path = 'pneumonia_model.keras'
    if not os.path.exists(model_path):
        print(f"ERROR: '{model_path}' not found. Run Trabajin.py first!")
        return

    print("Loading model... please wait...")
    model = tf.keras.models.load_model(model_path)
    print("Model loaded!")

    # 2. Open File Dialog to select image
    # This hides the main empty root window
    root = tk.Tk()
    root.withdraw()

    print("\nOpening file selector...")
    file_path = filedialog.askopenfilename(
        title="Select a Chest X-Ray",
        filetypes=[("Image files", "*.jpg *.jpeg *.png")]
    )

    if not file_path:
        print("No file selected. Exiting.")
        return

    print(f"Analyzing: {file_path}")

    # 3. Preprocess Image (Exact same way as training)
    IMG_SIZE = (150, 150)
    
    # Load and resize
    img = tf.keras.utils.load_img(file_path, target_size=IMG_SIZE)
    img_array = tf.keras.utils.img_to_array(img)
    
    # Create a batch (1, 150, 150, 3)
    img_batch = tf.expand_dims(img_array, 0)

    # 4. Predict
    # Note: Your model includes the Rescaling(1./255) layer, so we send raw pixels
    predictions = model.predict(img_batch)
    score = predictions[0][0]

    # 5. Visualize Result
    plt.figure(figsize=(6, 6))
    plt.imshow(img)
    
    # Logic: Output is sigmoid (0 to 1).
    # 0 = Class 0 (NORMAL), 1 = Class 1 (PNEUMONIA)
    if score > 0.5:
        confidence = score * 100
        plt.title(f"DIAGNOSIS: PNEUMONIA\nConfidence: {confidence:.2f}%", color='red', fontsize=14, fontweight='bold')
    else:
        confidence = (1 - score) * 100
        plt.title(f"DIAGNOSIS: NORMAL\nConfidence: {confidence:.2f}%", color='green', fontsize=14, fontweight='bold')
        
    plt.axis('off')
    
    # Remove the toolbar for a cleaner "App" look
    plt.gca().axes.get_xaxis().set_visible(False)
    plt.gca().axes.get_yaxis().set_visible(False)
    
    print("\nDisplaying result window...")
    plt.show()

if __name__ == "__main__":
    load_and_predict()