import tensorflow as tf
import sys
import os
from predict.needs.processData import *
from BurpyIC.settings import MODEL_DIR

graph_path = os.path.join(MODEL_DIR, 'drinks_graph.pb')
labels_path = os.path.join(MODEL_DIR, 'drinks_labels.txt')

tf.app.flags.DEFINE_string("output_graph", graph_path, "Path where the learned neural network is stored.")
tf.app.flags.DEFINE_string("output_labels", labels_path,"Label data files to learn.")
tf.app.flags.DEFINE_boolean("save_dict", True, "Save the image heuristic to dict.")
FLAGS = tf.app.flags.FLAGS

def inception_predict(image):
    labels = [line.rstrip() for line in tf.gfile.GFile(FLAGS.output_labels)]
    
    # Model restore using graphs(pb).
    with tf.gfile.FastGFile(FLAGS.output_graph, 'rb') as fp:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(fp.read())
        tf.import_graph_def(graph_def, name='')
    
    with tf.Session() as sess:
        logits = sess.graph.get_tensor_by_name('final_result:0')
        prediction = sess.run(logits,{'DecodeJpeg/contents:0':image})
    
    if FLAGS.save_dict:
        result = dict()
        for i in range(len(labels)):
            name = labels[i]
            rate = prediction[0][i] * 100
            result[name] = round(rate, 2)
        result = process_result_dict(result, 5)
        return result
    else:
        return None