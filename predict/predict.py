import tensorflow as tf
import numpy as np

from PIL import Image

from remindDjango.settings import MODEL_DIR
import os


def imageProcess(image):
    # greyscal and normalization
    im = image.convert('L')
    
    # Image resizing
    im = im.resize((28, 28), Image.ANTIALIAS)
    return im

def datalization(image):
    pixels = list(image.getdata())
    data = [(255 - x) * 1.0 / 255.0 for x in pixels]
    data = np.array(data)
    data = np.reshape(data, (1, 28, 28, 1))
    return data

def CNNprediction(data):
    # CNN dir settings
    save_path = os.path.join(MODEL_DIR, 'CNNmodel')

    X = tf.placeholder(tf.float32, [None, 28, 28, 1])
    keep_prob = tf.placeholder(tf.float32)
    
    # 1
    W1 = tf.Variable(tf.random_normal([3, 3, 1, 32], stddev=0.01), name = 'W1')
    L1 = tf.nn.conv2d(X, W1, strides=[1,1,1,1], padding='SAME')
    L1 = tf.nn.relu(L1)
    L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    
    # 2
    W2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01), name = 'W2')
    L2 = tf.nn.conv2d(L1, W2, strides=[1,1,1,1], padding='SAME')
    L2 = tf.nn.relu(L2)
    L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # 3
    W3 = tf.Variable(tf.random_normal([7 * 7 * 64, 256], stddev=0.01), name = 'W3')
    L3 = tf.reshape(L2, [-1, 7 * 7 * 64])
    L3 = tf.matmul(L3, W3)
    L3 = tf.nn.relu(L3)
    L3 = tf.nn.dropout(L3, keep_prob)

    # output
    W4 = tf.Variable(tf.random_normal([256, 10], stddev=0.01), name = 'W4')
    model = tf.matmul(L3, W4)

    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, save_path)
        predictions = sess.run(model, feed_dict={X: data, keep_prob: 0.7})
        result = sess.run(tf.nn.softmax(predictions[0]))
        result = result.argmax(axis=0)
    
    return result