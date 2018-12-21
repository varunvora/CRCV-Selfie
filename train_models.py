from common import *
from keras_models import get_model
from dataset_loader import img_loader

import pydot
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from keras.utils import plot_model
from matplotlib import pyplot as plt
import pickle
import os

def train_model(io: int):
    tensorboard_callback = TensorBoard(log_dir="logs",
                                       histogram_freq=0,
                                       write_graph=True,
                                       write_images=False)
    save_model_callback = ModelCheckpoint(os.path.join("weights", 'weights.' + str(io) + '.{epoch:02d}.h5'),
                                          verbose=3,
                                          save_best_only=True,
                                          save_weights_only=False,
                                          mode='auto',
                                          period=1)

    early_stopping_callback = EarlyStopping(monitor='val_loss',
                                            min_delta=0.001,
                                            patience=2,
                                            verbose=0, mode='auto')


    # BATCH_SIZE = 128
    # EPOCHS = 64
    # STEPS_PER_EPOCH = 512

    BATCH_SIZE = 1024
    EPOCHS = 8
    STEPS_PER_EPOCH = 32

    model = get_model(io=io, type="not resnet")
    print(model)

    if io == 1:
        history = model.fit_generator(
            img_loader(BATCH_SIZE, io=io),
            #     data_generator.batch_generator('train', batch_size=BATCH_SIZE),
            steps_per_epoch=STEPS_PER_EPOCH,
            epochs=EPOCHS,
            validation_data=img_loader(BATCH_SIZE, io=io, mode="test"),
            validation_steps=STEPS_PER_EPOCH,
            callbacks=[save_model_callback, tensorboard_callback, early_stopping_callback],

            #     workers=4,
            #     pickle_safe=True,
        )

        with open("pickled_history." + str(io) + ".pkl", "wb") as f:
            pickle.dump(history, f)
# Plot training & validation accuracy values
    plt.plot(history.history['mean_squared_error'])
    plt.plot(history.history['val_mean_squared_error'])
    plt.title('Model mean squared error')
    plt.ylabel('Mse')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig("train_metrics." + str(io) + ".png")

    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig("test_metrics." + str(io) + ".png")
