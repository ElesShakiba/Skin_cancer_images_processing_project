import numpy as np
import os
import cv2
from sklearn.model_selection import train_test_split

def load_data(data_dir, img_size):
    data = []
    labels = []
    categories = os.listdir(data_dir)
    for category in categories:
        path = os.path.join(data_dir, category)
        class_num = categories.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                resized_array = cv2.resize(img_array, img_size)
                data.append(resized_array)
                labels.append(class_num)
            except Exception as e:
                pass
    data = np.array(data).reshape(-1, img_size[0], img_size[1], 1)
    labels = np.array(labels)
    return data, labels

def preprocess(data_dir, img_size=(128, 128)):
    data, labels = load_data(data_dir, img_size)
    data = data / 255.0
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = preprocess("data/Skin cancer ISIC The International Skin Imaging Collaboration/Train")
