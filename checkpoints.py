import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import image as mpimg
from keras.applications.resnet50 import preprocess_input
from keras.preprocessing import image
import numpy as np
from common import *
from dataset_loader import img_loader


def analyze_dataset(dataset_file_name="./Selfie-dataset/selfie_dataset.txt"):
    global col
    select_col = ['female', 'baby', 'child', 'teenager', 'youth', 'middleAge', 'senior']

    data_df = pd.read_csv(dataset_file_name, sep="\s+", names=list(col.keys()))
    select_df = data_df[select_col]
    print("FIRST 10", select_df[:10], sep="\n")
    print("\n")
    print("LAST 10", select_df[-10:], sep="\n")

    for col in select_col:
        temp = select_df[col].value_counts()
        count = [0, 0, 0]
        count[0], count[2] = temp[-1], temp[1]
        if 0 in temp:
            count[1] = temp[0]
        df = pd.DataFrame({"x": [-1, 0, 1], "count": count})
        # ax = df.plot.bar(x="x", y="count", rot=0, title=col, )
        yield df, col

def plot_graph(history, metrics):
    fig = plt.figure(figsize=(15, 10))
    rows, columns = 2, len(metrics) # i think you want to change this
    for i, met in enumerate(metrics):
        fig.add_subplot(rows, columns, i + 1)
        plt.plot(history.history[met])
        plt.plot(history.history['val_' + met])
        plt.title('Model ' + met)
        plt.ylabel(met)
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()
    #done


age_group = {0: 'baby', 1: 'child', 2: 'teenager', 3: 'youth', 4: 'middle-age', 5: 'senior'}
quality = {0: 'poor', 1: 'average', 2: 'great'}


def get_img(img_path):
    img = image.load_img(img_path, target_size=IMAGE_DIM)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x[0]


def decode_output(op, num_samples, io=1):
    if io == 1:
        print('\nModel1: Gender and age predictions\n')
        fig = plt.figure(figsize=(15, 10))
        columns, rows = 5, 2
        for i in range(num_samples):
            g, a = op[0][i][0], op[1][i]
            gender = 'female' if round(g) == 1 else 'male'
            age = age_group[np.argmax(a)]
            img = mpimg.imread('samples/image' + str(i + 1) + '.jpg')
            fig.add_subplot(rows, columns, i + 1)
            plt.title(gender + ' ' + age)
            plt.imshow(img)
        plt.show()
    elif io == 2:
        print('\nModel2: Other attribute predictions\n')
        op = [list(map(lambda y: int(round(y)), x)) for x in op]
        print(*op, sep='\n')
    elif io == 3:
        print('\nModel3: Popularity score and other attributes\n')
        op[0] = ['Popularity score: ' + str(x[0]) for x in op[0]]
        op[1] = [list(map(lambda y: int(round(y)), x)) for x in op[1]]
        op = list(zip(op[0], op[1]))
        print(*op, sep='\n')
    elif io == 4:
        op = [x[0] for x in op]
        print('Actual:', op)
    elif io == 5:
        print('\nModel5: Quality of image\n')
        op = [quality[np.argmax(x)] for x in op]
        fig = plt.figure(figsize=(15, 10))
        columns, rows = 5, 2
        for i in range(num_samples):
            img = mpimg.imread('samples/image' + str(i + 1) + '.jpg')
            fig.add_subplot(rows, columns, i + 1)
            plt.title(op[i])
            plt.imshow(img)
        plt.show()
    elif io == 6:
        op = [quality[np.argmax(x)] for x in op]
        print('Actual:', op)


def try_model(model, num_samples, io=1):
    if io == 4:
        ip = next(img_loader(batch_size=num_samples, io=io))
        print('\nModel4: Popularity score from attributes\n')
        print('Expected:', ip[1])
        ip = ip[0]
    elif io == 6:
        ip = next(img_loader(batch_size=num_samples, io=io))
        print('\nModel6: Quality of selfie\n')
        print('Expected:', [quality[np.argmax(x)] for x in ip[1]])
        ip = ip[0]
    else:
        ip = []
        for i in range(num_samples):
            img_path = 'samples/image' + str(i + 1) + '.jpg'
            ip.append(get_img(img_path))
        ip = np.asarray(ip)
    op = model.predict(ip)
    decode_output(op, num_samples, io=io)


if __name__ == "__main__":
    analyze_dataset()

