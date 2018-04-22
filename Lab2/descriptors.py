import pickle
import numpy as np
from pprint import pprint
from matplotlib import pyplot as plt


def load_data(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data


def plot_picture(picture, title):
    plt.imshow(picture, cmap=plt.cm.gray)
    plt.title(title)
    plt.show()


if __name__ == "__main__":
    images = load_data("veeimgdump")
    target = load_data("veetrgtdump")
    print(np.shape(images))
    print(np.shape(target))
    print(np.unique(target))

    sample_image = images[180]
    print(np.shape(sample_image))
    #pprint(sample_image)
    #pprint(sample_image[185])
    #print(np.shape(sample_image[185]))
    plot_picture(sample_image, 'title')

