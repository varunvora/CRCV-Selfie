from common import *

import numpy as np
import pandas as pd
from keras.applications.resnet50 import preprocess_input
from keras.preprocessing import image
from sklearn.model_selection import train_test_split

def classify_popularity(popularityScore):
    quantiles = [3.989000, 4.379000, 4.768000]
    if(popularityScore <= quantiles[0]):
        return np.array([1, 0, 0])
    elif(popularityScore > quantiles[0] and popularityScore <= quantiles[1]):
        return np.array([0, 1, 0])
    else:
        return np.array([0, 0, 1])


def get_img(img_name: str):
    img_path = dataset_image_dir + img_name + ".jpg"
    img = image.load_img(img_path, target_size=IMAGE_DIM)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x[0]


def img_loader(batch_size: int, io: int, mode="train", split=0.15):
    data_df = pd.read_csv(dataset_file_name, sep=r"\s+")
    train_df, test_df = train_test_split(data_df, test_size=split)
    if mode == "train":
        data_list = train_df.as_matrix()
    else:  # mode == "test"
        data_list = test_df.as_matrix()
    del data_df, train_df, test_df  # explicitly clean up memory

    while True:
        batch_start = 0
        batch_end = batch_size

        while batch_start < len(data_list):
            limit = min(batch_end, len(data_list))

            if io == 1:  # the first model with image input and branched output of female and age
                X, female_output, age_output = [], [], []
                for i in range(batch_start, limit):
                    row = data_list[i]
                    X.append(get_img(row[col["imageName"]]))
                    female_output.append(np.array(row[col["female"]]))
                    age_output.append(np.array(row[4:10]))
                X = np.asarray(X)
                female_output = np.asarray(female_output)
                age_output = np.asarray(age_output)

                yield (X, [female_output, age_output])

            elif io == 2:  # the second model with image input and branched output to female and all other attributes
                X, Y = [], []
                for i in range(batch_start, limit):
                    row = data_list[i]
                    X.append(get_img(row[col["imageName"]]))
                    Y.append(np.array(row[2:]))  # all the columns except image name and popularity score
                X = np.asarray(X)
                Y = np.asarray(Y)

                yield (X, Y)

            elif io == 3:  # Given the image, predict the popularity score along with all the attributes
                X, popularity, Y = [], [], []
                for i in range(batch_start, limit):
                    row = data_list[i]
                    X.append(get_img(row[col["imageName"]]))
                    popularity.append(np.array(row[col["popularityScore"]]))
                    Y.append(np.array(row[2:]))
                X = np.asarray(X)
                popularity = np.asarray(popularity)
                Y = np.asarray(Y)

                yield (X, [popularity, Y])

            elif io == 4:  # Given the attributes predict the popularity score
                X, Y = [], []
                for i in range(batch_start, limit):
                    row = data_list[i]
                    X.append(np.asarray(row[2:]))
                    Y.append(np.array(row[col["popularityScore"]]))
                X = np.asarray(X)
                Y = np.asarray(Y)

                yield X, Y

            elif io == 5:
                X, Y = [], []
                for i in range(batch_start, limit):
                    row = data_list[i]
                    X.append(get_img(row[col["imageName"]]))
                    temp = row[col["popularityScore"]]
                    Y.append(classify_popularity(temp))
                X = np.asarray(X)
                Y = np.asarray(Y)

                yield X, Y

            elif io == 6:
                X, Y = [], []
                for i in range(batch_start, limit):
                    row = data_list[i]
                    X.append(np.asarray(row[2:]))
                    temp = row[col["popularityScore"]]
                    Y.append(classify_popularity(temp))
                X = np.asarray(X)
                Y = np.asarray(Y)

                yield X, Y

            batch_start += batch_size
            batch_end += batch_size


if __name__ == "__main__":
    for x_, y_ in img_loader(batch_size=1, mode="test", io=4):
        print(x_, y_)
        break
