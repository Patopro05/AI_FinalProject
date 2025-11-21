import os
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
np.set_printoptions(suppress=True)
def load_and_predict():
    model_path = 'pneumonia_model.keras'
    print("Loading model... please wait...")
    model = tf.keras.models.load_model(model_path)
    print("Model loaded!")
    current_dir = os.getcwd()
    sample_images_path = os.path.join(current_dir, '50SampleImages')
    root = tk.Tk()
    root.withdraw() 
    print("\nOpening file selector...")
    file_path = filedialog.askopenfilename(
        title="Select a Chest X-Ray",
        initialdir=sample_images_path,
        filetypes=[("Image files", "*.jpg *.jpeg *.png")])
    root.destroy() 
    if not file_path:
        print("No file selected. Exiting.")
        return
    print(f"Analyzing: {file_path}")
    IMG_SIZE = (150, 150)
    img = tf.keras.utils.load_img(file_path, target_size=IMG_SIZE)
    img_array = tf.keras.utils.img_to_array(img)
    img_batch = tf.expand_dims(img_array, 0)
    predictions = model.predict(img_batch)
    score = predictions[0][0]
    plt.figure(figsize=(6, 6))
    plt.imshow(img)
    if score > 0.5:
        confidence = score * 100
        plt.title(f"DIAGNOSIS: PNEUMONIA\nConfidence: {confidence:.2f}%", color='red', fontsize=14, fontweight='bold')
    else:
        confidence = (1 - score) * 100
        plt.title(f"DIAGNOSIS: NORMAL\nConfidence: {confidence:.2f}%", color='green', fontsize=14, fontweight='bold')
    plt.axis('off')
    plt.gca().axes.get_xaxis().set_visible(False)
    plt.gca().axes.get_yaxis().set_visible(False)
    print("\nDisplaying result window...")
    plt.show()
if __name__ == "__main__":
    load_and_predict()