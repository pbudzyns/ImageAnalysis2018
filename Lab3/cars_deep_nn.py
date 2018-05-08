from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import numpy as np
import tensorflow as tf
import pickle
from matplotlib import pyplot as plt

tf.logging.set_verbosity(tf.logging.INFO)

# Our application logic will be added here


def load_data(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data


def cnn_model_fn(features, labels, mode):
    """Model function for CNN."""
    # Input layer
    # TODO: change - batch, width, height, channels:3 for rgb
    # input_layer = tf.reshape(features["x"], [-1, 186, 250, 3])
    input_layer = features["x"]
    norm_input_layer = tf.nn.lrn(input_layer)

    conv1 = tf.layers.conv2d(
        inputs=norm_input_layer,
        filters=32,
        kernel_size=7,
        padding="same",
        activation=tf.nn.relu)

    # Pooling layer #1
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
    # norm1 = tf.nn.lrn(pool1, 4)


    # Convolutional Layer #2 and Pooling Layer #2
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=7,
        padding="same",
        activation=tf.nn.relu)
    norm2 = tf.nn.lrn(conv2)
    pool2 = tf.layers.max_pooling2d(inputs=norm2, pool_size=[2, 2], strides=2)
    """
    # Convolutional Layer #3 and Pooling Layer #3
    conv3 = tf.layers.conv2d(
        inputs=conv2,
        filters=512,
        kernel_size=11,
        padding="same",
        activation=tf.nn.relu)
    norm3 = tf.nn.lrn(conv3, 4)
    pool3 = tf.layers.max_pooling2d(inputs=norm3, pool_size=[4, 4], strides=2)
    """
    # Dense Layer
    # pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
    pool3_flat = tf.reshape(pool2, [-1, 182528])
    dense = tf.layers.dense(inputs=pool3_flat, units=20, activation=tf.nn.relu, use_bias=True)
    dropout = tf.layers.dropout(inputs=dense, rate=0.6, training=mode == tf.estimator.ModeKeys.TRAIN)

    # dense2 = tf.layers.dense(inputs=dropout, units=50, activation=tf.nn.relu)
    # dropout2 = tf.layers.dropout(inputs=dense2, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Logits
    # TODO: change to car type output
    cars = tf.layers.dense(inputs=dropout, units=3)

    predictions = {
        "classes": tf.argmax(input=cars, axis=1),
        "probabilities": tf.nn.sigmoid(cars, name="sigmoid_tensor")
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=3)
    loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=cars)
    # loss = tf.losses.sigmoid_cross_entropy(onehot_labels, logits=cars)
    # loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=cars)
    # loss = tf.losses.mean_squared_error(onehot_labels, cars)

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())

        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"])}

    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


if __name__ == "__main__":

    train_data = np.asanyarray(load_data('CARS-data/trainimgs'), dtype=np.float16)
    train_labels = np.asanyarray(load_data('CARS-data/traintrgt'), dtype=np.int32) - 2
    eval_data = np.asanyarray(load_data('CARS-data/testimgs'), dtype=np.float16)
    eval_labels = np.asanyarray(load_data('CARS-data/testtrgt'), dtype=np.int32) - 2

    # onehot_a = tf.one_hot(indices=tf.cast(eval_labels, tf.int32), depth=3)
    # sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
    #
    # print(sess.run(onehot_a))
    # tf.print_function(onehot_a)

    # print(np.shape(train_data))
    # print(np.shape(train_labels))
    # print(train_labels)
    # print(eval_labels)
    # print(train_data[1])

    # data2 = load_data('CARS-data/trainimgs')
    # def plot_picture(picture, title):
    #     plt.imshow(picture)
    #     plt.title(title)
    #     plt.show()
    #
    # for i in range(20):
    #     plot_picture(data2[i], train_labels[i])

    cars_classifier = tf.estimator.Estimator(model_fn=cnn_model_fn, model_dir="tmp/cars_convnet_model4")

    # Set up logging for predictions
    tensors_to_log = {"probabilities": "sigmoid_tensor"}
    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=50)

    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data},
        y=train_labels,
        batch_size=1,
        num_epochs=None,
        shuffle=True)

    # with tf.device('/cpu:0'):

    cars_classifier.train(
        input_fn=train_input_fn,
        steps=200,
        hooks=[logging_hook])
    #
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data},
        y=eval_labels,
        batch_size=1,
        num_epochs=1,
        shuffle=False)

    eval_results = cars_classifier.evaluate(input_fn=eval_input_fn)
    print(eval_results)
    

