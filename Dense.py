import tensorflow as tf
import os
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input
from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import csv

def create_densenet_model(input_shape, num_classes, learning_rate_schedule):
    input_tensor = Input(shape=input_shape)
    normalized_input = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)(input_tensor)
    three_channel_input = Concatenate(axis=-1)([normalized_input] * 3)
    base_model = DenseNet121(weights='imagenet', include_top=False, input_tensor=three_channel_input)
    x = GlobalAveragePooling2D()(base_model.output)
    predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=input_tensor, outputs=predictions)
    optimizer = Adam(learning_rate=learning_rate_schedule)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

train_data_directory = 'TrainDir'
validation_data_directory = 'ValDir'

batch_size = 32
epochs = 25
input_shape = (512, 512, 3) #have to adjust according to the dataset size of chakshu images
num_classes = 2  # for binary classification

# Data augmentation and preprocessing
train_datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    rescale=1./255
)

validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_data_directory,
    target_size=(512, 512),
    batch_size=batch_size,
    class_mode='binary'  # Set class_mode to 'binary' for binary classification
)

validation_generator = validation_datagen.flow_from_directory(
    validation_data_directory,
    target_size=(512, 512),
    batch_size=batch_size,
    class_mode='binary'  # Set class_mode to 'binary' for binary classification
)

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.001, decay_steps=10000, decay_rate=0.9, staircase=True
)

model = create_densenet_model(input_shape, num_classes, lr_schedule)

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_accuracy', patience=3, restore_best_weights=True
)

checkpoint_path = "model_checkpoint"
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    checkpoint_path, monitor='val_accuracy', save_best_only=True
)

#Prints the results; gets updated after each epoch
csv_file = 'training_logs.csv'

with open(csv_file, mode='w', newline='') as log_file:
    log_writer = csv.writer(log_file)
    log_writer.writerow(['Epoch', 'Loss', 'Accuracy'])

    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=validation_generator,
        callbacks=[early_stopping, model_checkpoint],
    )

    for epoch, metrics in enumerate(history.history, start=1):
        train_loss, train_accuracy = metrics['loss'], metrics['accuracy']
        val_loss, val_accuracy = metrics['val_loss'], metrics['val_accuracy']
        log_writer.writerow([epoch, train_loss, train_accuracy])
        log_writer.writerow([epoch, val_loss, val_accuracy])

tf.saved_model.save(model, 'densenet_saved_model')
