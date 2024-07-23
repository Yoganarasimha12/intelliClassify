import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping
from utils.data_loader import load_data
from models.model import build_model

def scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return float(lr * tf.math.exp(-0.1))

def train_model(dataset_path, save_path):
    input_shape = (64, 64, 3)
    num_classes = 10

    model = build_model(input_shape, num_classes)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    train_generator, validation_generator, _ = load_data(dataset_path, batch_size=32)

    lr_schedule = LearningRateScheduler(scheduler)
    early_stopping = EarlyStopping(monitor='val_loss', patience=5)

    model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=50,
        callbacks=[lr_schedule, early_stopping]
    )

    model.save(save_path)

if __name__ == "__main__":
    dataset_path = "C:/Users/naras/PycharmProjects/Satellite_mini/data/2750"
    save_path = "C:/Users/naras/PycharmProjects/Satellite_mini/models/trained_model.h5"
    train_model(dataset_path, save_path)
