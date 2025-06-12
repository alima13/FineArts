import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_data(image_size, batch_size):
    data_path = 'data/savoias_dataset'
    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

    train_gen = datagen.flow_from_directory(
        data_path,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training'
    )
    
    val_gen = datagen.flow_from_directory(
        data_path,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation'
    )
    
    class_names = list(train_gen.class_indices.keys())
    return train_gen, val_gen, class_names
