import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from model import model_fn
import time
import argparse
import os
tf.logging.set_verbosity(tf.logging.INFO)

parser = argparse.ArgumentParser(description='check if got existing checpoint to resume from')
parser.add_argument('--checkpoint', default=None)
args = parser.parse_args()
print(args)

mnist = input_data.read_data_sets("./data/fashion", one_hot=False)
print("There are %s training images, %s validation images, and %s test images"%(len(mnist.train.images),len(mnist.validation.images),len(mnist.test.images)))
def train_input_fn():
    dataset = tf.data.Dataset.from_tensor_slices(
    (mnist.train.images, mnist.train.labels.astype(np.int32)))
    dataset = dataset.shuffle(
    buffer_size=1000, reshuffle_each_iteration=True).repeat(count=None).batch(128)
    train_iterator = dataset.make_one_shot_iterator()
    features, labels = train_iterator.get_next()
    return features, labels

def validation_input_fn():
    dataset = tf.data.Dataset.from_tensor_slices(
    (mnist.train.images, mnist.train.labels.astype(np.int32)))
    dataset = dataset.batch(128)
    val_iterator = dataset.make_one_shot_iterator()
    features, labels = val_iterator.get_next()
    return features, labels

# config

CHECKPOINT=args.checkpoint
# CHECKPOINT = './model_trained/model.ckpt-30000'
STEPS = 30000 
BATCH = 128
LEARNING_RATE = 0.01
model_params = {"learning_rate": LEARNING_RATE}
est_config = tf.estimator.RunConfig()
est_config = est_config.replace(
        keep_checkpoint_max=10,
        save_checkpoints_steps=mnist.train.num_examples/BATCH,
        save_summary_steps=mnist.train.num_examples/BATCH)
if CHECKPOINT:
    model_dir = os.path.dirname(CHECKPOINT)
    estimator = tf.estimator.Estimator(model_fn=model_fn, params=model_params, model_dir =model_dir, warm_start_from =CHECKPOINT)
else:
    model_dir = './model'
    estimator = tf.estimator.Estimator(model_fn=model_fn, params=model_params, model_dir =model_dir)

# Run training and validation in between, this is quite slow since every time it started a new graph. 
# for _ in range(30):
#     estimator.train(input_fn=train_input_fn, steps = 1000)
#     metrics = estimator.evaluate(input_fn=validation_input_fn)

start = time.time()
estimator.train(input_fn=train_input_fn, steps=STEPS)
print("Finished training process! It takes %s ms"%((time.time()-start)*1000))

# Evaluation on validation dataset and testing dataset
val_results = estimator.evaluate(input_fn=validation_input_fn)
print("\nValidation accuracy: %g %%" % (val_results["accuracy"]*100))
def test_input_fn():
    dataset = tf.data.Dataset.from_tensor_slices(
    (mnist.test.images, mnist.test.labels.astype(np.int32)))
    dataset = dataset.batch(BATCH)
    test_iterator = dataset.make_one_shot_iterator()
    features, labels = test_iterator.get_next()
    return features, labels
test_results = estimator.evaluate(input_fn=test_input_fn)
accuracy_score = test_results["accuracy"]
print("\nTest accuracy: %g %%" % (accuracy_score*100))






