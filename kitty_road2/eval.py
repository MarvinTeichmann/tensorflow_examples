"""Evaluation of the Model.


"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import math
import time

import tensorflow.python.platform
from tensorflow.python.platform import gfile
import numpy as np
import tensorflow as tf

import utils

import data_input as data_input
import smallnet as model
import os


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('eval_dir', 'eval',
                           """Directory where to write event logs, relative to checkpoint_dir.""")
tf.app.flags.DEFINE_string('eval_data', 'test',
                           """Either 'test' or 'train_eval'.""")

# TODO: Iterate over all possible Values
# Write Values to Tensorboard


def do_eval(sess,
            eval_correct,
            keep_prob):
  """Runs one evaluation against the full epoch of data.

  Args:
    sess: The session in which the model has been trained.
    eval_correct: The Tensor that returns the number of correct predictions.
    keep_prob: The keep prob placeholder.
  """
  # And run one epoch of eval.
  true_count = 0  # Counts the number of correct predictions.
  steps_per_epoch = data_input.NUM_EXAMPLES_PER_EPOCH_FOR_TEST // FLAGS.batch_size
  num_examples = steps_per_epoch * FLAGS.batch_size
  for step in xrange(steps_per_epoch):
    feed_dict = utils.fill_feed_dict(keep_prob,
                                     train = False)
    true_count += sess.run(eval_correct, feed_dict=feed_dict)
  precision = true_count / num_examples
  print('  Num examples: %d  Num correct: %d  Precision @ 1: %0.04f' %
        (num_examples, true_count, precision))

def evaluate_last():
  """Loads the model and runs evaluation
  """

  with tf.Graph().as_default():
    # Get images and labels for CIFAR-10.
    model_dir = os.path.join(FLAGS.model_dir,FLAGS.name)

    eval_data = FLAGS.eval_data == 'test'
    images, labels =  data_input.inputs(eval_data=eval_data, data_dir=FLAGS.data_dir,
                                           batch_size=FLAGS.batch_size)

    #images, labels =  data_input.distorted_inputs(eval_data=eval_data, data_dir=FLAGS.data_dir,
    #                                       batch_size=FLAGS.batch_size)

    # Generate placeholders for the images and labels.
    keep_prob = utils.placeholder_inputs(FLAGS.batch_size)

    # Build a Graph that computes predictions from the inference model.
    logits = model.inference(images, keep_prob)

    # Add to the Graph the Ops for loss calculation.
    loss = model.loss(logits, labels)
  
    # Calculate predictions.
    top_k_op = tf.nn.in_top_k(logits, labels, 1)

    # Add the Op to compare the logits to the labels during evaluation.
    eval_correct = model.evaluation(logits, labels)

    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.merge_all_summaries()

    # Create a saver for writing training checkpoints.
    saver = tf.train.Saver()

    # Create a session for running Ops on the Graph.
    sess = tf.Session()
  
    # Restore the moving average version of the learned variables for eval.
    # variable_averages = tf.train.ExponentialMovingAverage(
    #     cifar10.MOVING_AVERAGE_DECAY)
    # variables_to_restore = variable_averages.variables_to_restore()
    # saver = tf.train.Saver(variables_to_restore)
    # Build the summary operation based on the TF collection of Summaries.
    # summary_op = tf.merge_all_summaries()

    #graph_def = tf.get_default_graph().as_graph_def()
    #summary_writer = tf.train.SummaryWriter(FLAGS.eval_dir,
    #	                                     graph_def=graph_def)

    # Run the Op to initialize the variables.
    init = tf.initialize_all_variables()
    sess.run(init)

    # Start the queue runners.
    tf.train.start_queue_runners(sess=sess)

    print(model_dir)

    ckpt = tf.train.get_checkpoint_state(model_dir)
    if ckpt and ckpt.model_checkpoint_path:
      saver.restore(sess, ckpt.model_checkpoint_path)
    else:
      print("No checkpoints found! ")
      exit(1)

    print("Doing Evaluation with lots of data")  
    utils.do_eval(sess=sess,
                  eval_correct=eval_correct,
                  keep_prob=keep_prob, 
                  num_examples=data_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL)




def main(_):
  if(FLAGS.name is None):
    print("Usage: eval.py --name=NAME")
    exit(1)
  evaluate_last()




if __name__ == '__main__':
  tf.app.run()
