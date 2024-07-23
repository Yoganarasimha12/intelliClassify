from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_data(data_dir, batch_size=32):
    datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=0.2  # 20% of data for validation
    )

    train_generator = datagen.flow_from_directory(
        data_dir,
        target_size=(64, 64),
        batch_size=batch_size,
        class_mode='sparse',
        subset='training'
    )

    validation_generator = datagen.flow_from_directory(
        data_dir,
        target_size=(64, 64),
        batch_size=batch_size,
        class_mode='sparse',
        subset='validation'
    )

    test_datagen = ImageDataGenerator(rescale=1.0 / 255)

    test_generator = test_datagen.flow_from_directory(
        data_dir,
        target_size=(64, 64),
        batch_size=batch_size,
        class_mode='sparse'
    )

    return train_generator, validation_generator, test_generator
