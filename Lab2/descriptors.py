import pickle
import gist
import numpy as np
from pprint import pprint
from matplotlib import pyplot as plt
from scipy import signal
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# Tags: 0, 1-regular car, 2-ambulance


def load_data(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data


def plot_picture(picture, title):
    plt.imshow(picture)
    plt.title(title)
    plt.show()


def get_gist_descriptor(image):
    return gist.extract(image)


def get_images_descriptions(images):
    print('Computing images descriptions.... ')
    desc_size = len(get_gist_descriptor(images[0]))
    result = np.zeros((len(images), desc_size))
    for i, img in enumerate(images):
        result[i] = get_gist_descriptor(img)
    return result


def split_data(descriptions, targets, percent):
    return train_test_split(descriptions, targets, test_size=percent)


def autocorr(data):
    result = np.correlate(data, data, mode='full')
    size = len(result)
    idx = int(size/2) if not size//2 else int((size+1)/2)

    return result[idx:]


def find_peak(data):
    peakid = signal.find_peaks_cwt(data, np.arange(200, 300))
    return min(peakid)


def get_mean_periodicity(descriptions):
    results = np.zeros(len(descriptions))
    for i, desc in enumerate(descriptions):
        # description = get_gist_descriptor(img)
        results[i] = find_peak(desc)
    return int(np.mean(results))


def reduce_data_by_periodicity(descriptions):
    # mean_peak = get_mean_periodicity(descriptions) #Result: 159
    print('Data reduction by mean periodicity....')
    mean_peak = 159
    result = descriptions[:, :mean_peak]
    return result


def reduce_data_by_window_mean(descriptions, windows=10):
    print('Data reduction by window mean....')
    # size = len(descriptions)
    # third_part = int(size/3)
    results = np.zeros((len(descriptions), windows*3))
    for i, desc in enumerate(descriptions):
        parts = np.split(desc, 3)
        res = np.array([np.mean(np.split(part, windows), 1) for part in parts])
        results[i] = res.flatten()
    return results


def train_svm(train_data, train_targets):
    print('SVC training....')
    clf = SVC()
    clf.fit(train_data, train_targets)
    return clf


def test_svm_classification(descriptions, targets, test_percent=0.1):
    train_desc, test_desc, train_targets, test_target = split_data(descriptions, targets, test_percent)
    classifier = train_svm(train_desc, train_targets)

    print('SVC testing .....')
    size = len(test_desc)
    results = np.zeros(size)
    idxs = np.zeros(size)
    for i in range(size):
        prediction = classifier.predict([test_desc[i]])
        results[i] = prediction == test_target[i]
        # plot_picture(imgs[i], 'Target: %d, predicted: %d'%(test_target[i], prediction))
    print(np.mean(results))


def plot_data_compare(data1, data2):
    fig = plt.figure()
    ax = fig.add_subplot(121)
    ax.imshow(data1)
    # ax.plot(data1)
    ax.set_title('Cutted data')

    ax2 = fig.add_subplot(122)
    ax2.imshow(data2)
    # ax2.plot(data2)
    ax2.set_title('Whole description')
    plt.show()


if __name__ == "__main__":
    images = load_data("veeimgdump")
    targets = load_data("veetrgtdump")
    # descriptions = get_images_descriptions(images)
    descriptions = load_data("pictures_desctiptions")
    # print(np.shape(images))
    # print(np.shape(targets))
    # print(np.unique(targets))

    #pprint(sample_image)
    #pprint(sample_image[185])
    #print(np.shape(sample_image[185]))
    # descriptor = gist.extract(sample_image)
    # print(descriptor)
    #plot_picture(descriptor, 'Descriptor')
    # reduced = descriptor[310:620]
    # plt.plot(reduced)
    # plt.plot(descriptor)
    # plt.show()

    # img = images[0]
    # gray_pic = np.mean(img, axis=2)
    # plot_picture(gray_pic, 'Gray')
    # print(np.shape(img))
    # print(np.shape(gray_pic))
    # print(len(get_gist_descriptor(gray_pic)))



    # descriptions = reduce_data_by_periodicity(descriptions)
    descriptions = reduce_data_by_window_mean(descriptions, windows=5)

    for i in range(5):
        test_svm_classification(descriptions, targets, test_percent=0.1)
# for i in range(1, 5):
        # idx = i
        # sample_image = images[idx]
        # sample_target = target[idx]
        # description = get_gist_descriptor(sample_image)
        # peakid = find_peak(description)

        # plt.plot(description[:peakid])
        # plt.show()
        # plot_data_compare(descriptions[i], descriptions[i][:mean_peak])
