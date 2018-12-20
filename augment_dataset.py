from random import randrange
from random import choice, random, sample
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pydot
from sklearn.model_selection import train_test_split

from keras.applications.resnet50 import preprocess_input, decode_predictions, ResNet50

# DATASET COLUMNS
col = {
    'imageName': 0,
    'popularityScore': 1,
    'partialFaces': 2,
    'female': 3,
    'baby': 4,
    'child': 5,
    'teenager': 6,
    'youth': 7,
    'middleAge': 8,
    'senior': 9,
    'white': 10,
    'black': 11,
    'asian': 12,
    'ovalFace': 13,
    'roundFace': 14,
    'heartFace': 15,
    'smiling': 16,
    'mouthOpen': 17,
    'frowning': 18,
    'wearingGlasses': 19,
    'wearingSunglasses': 20,
    'wearingLipstick': 21,
    'tongueOut': 22,
    'duckFace': 23,
    'blackHair': 24,
    'blondHair': 25,
    'brownHair': 26,
    'redHair': 27,
    'curlyHair': 28,
    'straightHair': 29,
    'braidHair': 30,
    'showingCellphone': 31,
    'usingEarphone': 32,
    'usingMirror': 33,
    'braces': 34,
    'wearingHat': 35,
    'harshLighting': 36,
    'dimLighting': 37
}

datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)


def get_img(img_path, transform=False):
    img_path = "./Selfie-dataset/images/" + str(img_path) + ".jpg"
    img = image.load_img(img_path, target_size=(306, 306))
    x = image.img_to_array(img)

    if transform:
        x = datagen.random_transform(x, seed=randrange(1, 1000))

    return image.array_to_img(x)


if __name__ == "__main__":
    df = pd.read_csv('Selfie-dataset/selfie_dataset.txt', header=None, delim_whitespace=True)
    df = df.replace(to_replace=-1, value=0)
    CLASS_SIZE = 10000
    feature_dict = {
        'female': max(0, CLASS_SIZE - 33655),
        'baby': max(0, CLASS_SIZE - 196),
        'child': max(0, CLASS_SIZE - 796),
        'teenager': max(0, CLASS_SIZE - 6275),
        'youth': max(0, CLASS_SIZE - 31644),
        'middleAge': max(0, CLASS_SIZE - 1119),
        'senior': max(0, CLASS_SIZE - 16)
    }
    aug_df = []
    for feature in feature_dict.keys():
        cond = df[col[feature]] == 1
        sub_df = df[cond].as_matrix()

        rand_idxs = sample(range(len(sub_df)), min(CLASS_SIZE, len(sub_df)))

        for row in sub_df[rand_idxs]:
            # get_img(row[0]).save("./Augmented-Selfie-dataset/images/" + str(row[0]) + "_" + ".jpg")
            aug_df.append(row)

        for i in range(feature_dict[feature]):
            rand_idx = i % len(sub_df)
            aug_row = list(sub_df[rand_idx])
            aug_row[0] = "aug_" + str(i) + "_" + feature + "_" + aug_row[0]
            get_img(sub_df[rand_idx][0], transform=True).save(
                "./Augmented-Selfie-dataset/images/" + str(aug_row[0]) + ".jpg")
            aug_df.append(aug_row)

        print(len(aug_df))

    aug_df = pd.DataFrame(aug_df)
    aug_df.to_csv('./Augmented-Selfie-dataset/selfie-dataset.txt', sep=' ', index=False, header=None)

