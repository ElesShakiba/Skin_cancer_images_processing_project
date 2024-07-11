from tensorflow.keras.models import load_model
from preprocess import preprocess

def evaluate_model(model_path, data_dir):
    model = load_model(model_path)
    X_train, X_test, y_train, y_test = preprocess(data_dir)
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    evaluate_model("models/skin_cancer_classification_model.h5", "data/Skin cancer ISIC The International Skin Imaging Collaboration/Test")
