import tensorflow as tf
from utils.data_loader import load_data
from models.model import build_model

def evaluate_model(dataset_path, model_path):
    _, _, test_generator = load_data(dataset_path, batch_size=32)

    model = tf.keras.models.load_model(model_path)

    evaluation = model.evaluate(test_generator)
    print("Evaluation results:", evaluation)

if __name__ == "__main__":
    dataset_path = "C:/Users/naras/PycharmProjects/Satellite_mini/data/2750"
    model_path = "C:/Users/naras/PycharmProjects/Satellite_mini/models/trained_model.h5"
    evaluate_model(dataset_path, model_path)
