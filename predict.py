import cv2
import numpy as np
from tensorflow.keras.models import load_model

def predict_image(model_path, image_path, img_size=(128, 128)):
    model = load_model(model_path)
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, img_size)
    img = img.reshape(1, img_size[0], img_size[1], 1) / 255.0
    prediction = model.predict(img)
    return np.argmax(prediction, axis=1)  # Return the index of the highest probability

if __name__ == "__main__":
    result = predict_image("models/skin_cancer_classification_model.h5", "data/Skin cancer ISIC The International Skin Imaging Collaboration/Test/img1.jpg")
    print(f"Prediction: {result}")
