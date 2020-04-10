import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from model import model_fn
tf.logging.set_verbosity(tf.logging.INFO)

mnist = input_data.read_data_sets("./data/fashion", one_hot=False)
STEPS = 30000
estimator = tf.estimator.Estimator(model_fn=model_fn, model_dir ='./model_trained')
BATCH = 128

predict_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"image": mnist.test.images[:10]},
    num_epochs=1,
    shuffle=False)

def test_input_fn():
    dataset = tf.data.Dataset.from_tensor_slices(
    (mnist.test.images, mnist.test.labels.astype(np.int32)))
    dataset = dataset.batch(BATCH)
    test_iterator = dataset.make_one_shot_iterator()
    features, labels = test_iterator.get_next()
    return features, labels

predictions = estimator.predict(input_fn=test_input_fn)
for i, p in enumerate(predictions):
    print("Prediction %s: %s" % (i+1, p["class_ids"]))