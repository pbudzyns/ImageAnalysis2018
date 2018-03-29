import numpy as np
from scipy import misc
import glob
from matplotlib import pyplot as plt


def read_pictures(directory):
    type = 'bmp'
    pictures, tests = {}, {}
    filenames = glob.glob(directory + '*.' + type)
    for filename in filenames:
        name = filename.replace(directory, '').replace('.' + type, '')
        if 'test' not in filename:
            pictures[name] = misc.imread(filename, mode='L')
        elif 'test' in filename:
            tests[name] = misc.imread(filename, mode='L')

    return pictures, tests


def compute_potential_matrix(picture):
    m, n = np.shape(picture)
    help_matrix = np.zeros((m+2, n+2))
    help_matrix[1:n+1, 1:m+1] = picture[:, :]/255
    potential_matrix = np.zeros((m, n))

    for x in range(1, m+1):
        for y in range(1, n+1):
            potential = np.sum(help_matrix[x-1:x+2, y-1:y+2]) - help_matrix[x, y]
            # print(help_matrix[x-1:x+2, y-1:y+2], potential)
            # print(potential)
            potential_matrix[x-1, y-1] = potential/2 + help_matrix[x, y]

    return potential_matrix


def prepare_potential_map(pictures):
    pictures_potentials = {}
    for key, pic in pictures.items():
        pictures_potentials[key] = compute_potential_matrix(pic)

    return pictures_potentials


def get_distance_between_pics(first_picture, second_picture):
    if np.shape(first_picture) != np.shape(second_picture):
        return None

    distance = 0
    m, n = np.shape(first_picture)
    for x in range(m):
        for y in range(n):
            distance += (first_picture[x, y] - second_picture[x, y])**2

    return np.sqrt(distance)


def make_prediction(picture, dataset_potentials):
    results = {}
    picture_pot = compute_potential_matrix(picture)
    for key, potential in dataset_potentials.items():
        results[key] = get_distance_between_pics(picture_pot, potential)

    return min(results, key=results.get)


def plot_picture(picture):
    plt.imshow(picture, cmap=plt.cm.gray)
    plt.show()


if __name__ == "__main__":
    pictures, tests = read_pictures('pic/')

    pictures_potentials = prepare_potential_map(pictures)

    for key, pic in tests.items():
        prediction = make_prediction(pic, pictures_potentials)
        print(key, ': ', prediction)
