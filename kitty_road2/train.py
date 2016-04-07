"""Trains, Evaluates and Saves the model network using a Queue."""
# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import time
import json

import tensorflow.python.platform
import numpy
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

import utils as utils


import data_input as data_input
import smallnet as model


flags = tf.app.flags
FLAGS = flags.FLAGS


def run_training():
  """Train model for a number of steps."""
  # Get the sets of images and labels for training, validation, and
  # test on MNIST.

  # Tell TensorFlow that the model will be built into the default Graph.
  train_dir = os.path.join(FLAGS.model_dir,FLAGS.name)

  with open('kitti.json', 'r') as f:
    hypes = json.load(f)

  with tf.Graph().as_default():

    global_step = tf.Variable(0, trainable=False)


    with tf.name_scope('Input'):
      q = data_input.create_queues(hypes)
      image_batch, label_batch = data_input.inputs(q, 'train', hypes['solver']['batch_size'])

      # Generate placeholders for the images and labels.
      keep_prob = utils.placeholder_inputs()

    # Build a Graph that computes predictions from the inference model.
    logits = model.inference(image_batch, keep_prob)

    # Add to the Graph the Ops for loss calculation.
    loss = model.loss(logits, label_batch)

    # Add to the Graph the Ops that calculate and apply gradients.
    train_op = model.training(loss, global_step=global_step, learning_rate=FLAGS.learning_rate)

    # Add the Op to compare the logits to the labels during evaluation.
    eval_correct = model.evaluation(logits, label_batch)

    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.merge_all_summaries()

    # Create a saver for writing training checkpoints.
    saver = tf.train.Saver()

    # Create a session for running Ops on the Graph.
    sess = tf.Session()

    # Run the Op to initialize the variables.
    init = tf.initialize_all_variables()
    sess.run(init)

    # Start the queue runners.
    tf.train.start_queue_runners(sess=sess)

    data_input.start_enqueuing_threads(hypes, q, sess)

    # Instantiate a SummaryWriter to output summaries and the Graph.
    summary_writer = tf.train.SummaryWriter(train_dir,
                                            graph_def=sess.graph_def)

    # And then after everything is built, start the training loop.
    for step in xrange(FLAGS.max_steps):
      start_time = time.time()

      # Fill a feed dictionary with the actual set of images and labels
      # for this particular training step.
      feed_dict = utils.fill_feed_dict(keep_prob,
                                       train = True)

      # Run one step of the model.  The return values are the activations
      # from the `train_op` (which is discarded) and the `loss` Op.  To
      # inspect the values of your Ops or variables, you may include them
      # in the list passed to sess.run() and the value tensors will be
      # returned in the tuple from the call.
      _, loss_value = sess.run([train_op, loss],
                               feed_dict=feed_dict)

      # Write the summaries and print an overview fairly often.
      if step % 100 == 0:
        # Print status to stdout.
        duration = time.time() - start_time
        examples_per_sec = hypes['solver']['batch_size'] / duration
        sec_per_batch = float(duration)
        print('Step %d: loss = %.2f ( %.3f sec (per Batch); %.1f examples/sec;)' % (step, loss_value,
                                     sec_per_batch, examples_per_sec))
        # Update the events file.
        summary_str = sess.run(summary_op, feed_dict=feed_dict)
        summary_writer.add_summary(summary_str, step)

      # Save a checkpoint and evaluate the model periodically.
      if (step + 1) % 1000 == 0 or (step + 1) == FLAGS.max_steps:
        checkpoint_path = os.path.join(train_dir, 'model.ckpt')
        saver.save(sess, checkpoint_path , global_step=step)
        # Evaluate against the training set.

      if (step + 1) % 10000 == 0 or (step + 1) == FLAGS.max_steps:  
        print('Training Data Eval:')
        utils.do_eval(sess,
                      eval_correct,
                      keep_prob,
                      data_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN)
        # Evaluate against the validation set.
        # TODO!
        # Evaluate against the test set.
        # TODO!

def main(_):
  if(FLAGS.name is None):
    print("Usage: train.py --name=NAME")
    exit(1)
  run_training()


if __name__ == '__main__':
  tf.app.run()
