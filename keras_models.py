from common import *

import os
import pickle
from keras.models import load_model, Model, Sequential
from keras.layers import Flatten, Activation, Dropout, Dense, BatchNormalization, Input, Conv2D, MaxPool2D
from keras.applications.resnet50 import preprocess_input, decode_predictions, ResNet50
import pydot
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from keras.utils import plot_model
from matplotlib import pyplot as plt

def get_model(io: int, type="resnet"):

    if type == "resnet":

        model_base = ResNet50(include_top=False, input_shape=(*IMAGE_DIM, 3), weights='imagenet')
        output = Flatten()(model_base.output)
        # output = BatchNormalization()(output)
        # output = Dropout(0.5)(output)
        # output = Dense(128, activation='relu')(output)
        # output = BatchNormalization()(output)
        # output = Dropout(0.5)(output)
    else:
        input_base = Input(shape=(*IMAGE_DIM, 3))
        output = Conv2D(filters=16, kernel_size=(3, 3), activation='relu')(input_base)
        output = MaxPool2D(pool_size=(3, 3))(output)
        output = Dropout(rate=0.5)(output)
        output = Flatten()(output)
        output = Dense(128, activation='relu')(output)
        output = Dropout(0.5)(output)

    if io == 1:
        female_model = Dense(64, activation='relu')(output)
        female_model = Dropout(0.5)(female_model)
        female_model = Dense(8, activation='relu')(female_model)
        female_model = Dense(1, activation='sigmoid', name="Female")(female_model)

        age_model = Dense(64, activation='relu')(output)
        age_model = Dropout(0.5)(age_model)
        age_model = Dense(8, activation='relu')(age_model)
        age_model = Dense(6, activation='softmax', name="Age")(age_model)

        if type == "resnet":
            model = Model(input=model_base.input, output=[female_model, age_model])
            for layer in model_base.layers:
                layer.trainable = False
        else:
            model = Model(input=input_base, output=[female_model, age_model])
        losses = {
            "Female": "binary_crossentropy",
            "Age": "categorical_crossentropy"
        }


    elif io == 2:
        all_attr = Dense(64, activation='relu')(output)
        all_attr = Dropout(0.5)(all_attr)
        all_attr = Dense(36, activation='sigmoid', name="Attributes")(all_attr)

        if type == "resnet":
            model = Model(input=model_base.input, output=all_attr)
            for layer in model_base.layers:
                layer.trainable = False
        else:
            model = Model(input=input_base, output=all_attr)


        losses = {
            "Attributes": "binary_crossentropy"
        }

    elif io == 3:
        popularity_model = Dense(64, activation='relu')(output)
        popularity_model = Dropout(0.5)(popularity_model)
        popularity_model = Dense(8, activation='relu')(popularity_model)
        popularity_model = Dense(1, activation='relu', name="Popularity")(popularity_model)

        all_attr = Dense(64, activation='relu')(output)
        all_attr = Dropout(0.5)(all_attr)
        all_attr = Dense(36, activation='sigmoid', name="Attributes")(all_attr)

        if type == "resnet":
            model = Model(input=model_base.input, output=[popularity_model,all_attr])
            for layer in model_base.layers:
                layer.trainable = False
        else:
            model = Model(input=input_base, output=[popularity_model,all_attr])

        losses = {
            "Popularity": "mse",
            "Attributes": "binary_crossentropy"
        }



    elif io == 4:
        model = Sequential()
        model.add(Dense(512, input_shape=(36,), activation='sigmoid'))
        # model.add(Dense(128, activation='sigmoid'))
        # model.add(Dense(32, activation='sigmoid'))
        model.add(Dropout(0.5))
        model.add(Dense(8, activation='relu'))
        model.add(Dense(1, activation='relu', name="Popularity"))

        losses = {
            "Popularity": "mse"
        }

    model.compile(optimizer='adam',
                  loss=losses,
                  metrics= ['acc', 'mse', 'mae'])


    model.summary()

    # Generate a plot of a model
    # pydot.find_graphviz = lambda: True
    try:
        plot_model(model, show_shapes=True, to_file='model_image.'+ str(io) + '.png')
    except:
        print("No graphviz to print the model's image")

    return model

if __name__ == "__main__":
    for i in range(1, 5):
        # if i != 4:
        #     continue
        tensorboard_callback = TensorBoard(log_dir="logs",
                                           histogram_freq=0,
                                           write_graph=True,
                                           write_images=False)
        save_model_callback = ModelCheckpoint(os.path.join("weights", 'weights.' + str(i) + '.{epoch:02d}.h5'),
                                              verbose=3,
                                              save_best_only=True,
                                              save_weights_only=False,
                                              mode='auto',
                                              period=1)

        early_stopping_callback = EarlyStopping(monitor='val_loss',
                                                min_delta=0.001,
                                                patience=2,
                                                verbose=0, mode='auto')


        BATCH_SIZE = 128
        EPOCHS = 64
        STEPS_PER_EPOCH = 512

        # BATCH_SIZE = 1
        # EPOCHS = 1
        # STEPS_PER_EPOCH = 1

        model = get_model(io=i, type="not resnet")
        print(model)
        from dataset_loader import img_loader
        history = model.fit_generator(
            img_loader(BATCH_SIZE, io=i),
            #     data_generator.batch_generator('train', batch_size=BATCH_SIZE),
            steps_per_epoch=STEPS_PER_EPOCH,
            epochs=EPOCHS,
            validation_data=img_loader(BATCH_SIZE, io=i, mode="test"),
            validation_steps=STEPS_PER_EPOCH,
            callbacks=[save_model_callback, tensorboard_callback, early_stopping_callback],

            #     workers=4,
            #     pickle_safe=True,
        )

        with open("pickled_history." + str(i) + ".pkl", "wb") as f:
            pickle.dump(history, f)
        continue
    # Plot training & validation accuracy values
        plt.plot(history.history['mean_squared_error'])
        plt.plot(history.history['val_mean_squared_error'])
        plt.title('Model mean squared error')
        plt.ylabel('Mse')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.savefig("train_metrics." + str(i) + ".png")

        # Plot training & validation loss values
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.savefig("test_metrics." + str(i) + ".png")
