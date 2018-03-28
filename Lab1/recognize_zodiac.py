import scipy
import numpy as np
from scipy import misc
import glob


def read_pictures(directory):
    pictures, tests = {}, {}
    filenames = glob.glob(directory + '*.bmp')
    for filename in filenames:
        name = filename.replace(directory, '').replace('.bmp', '')
        if 'test' not in filename:
            pictures[name] = misc.imread(filename, mode='L')
        elif 'test' in filename:
            tests[name] = misc.imread(filename, mode='L')

    return pictures, tests



if __name__ == "__main__":
    pictures, tests = read_pictures('pic/')
    print(pictures.keys())
    print(tests.keys())

    print(np.shape(pictures['aries']))