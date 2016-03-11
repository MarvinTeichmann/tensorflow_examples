from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import sys
import tensorflow as tf

import params

def training(loss, global_step=0, learning_rate=None):
  """Sets up the training Ops.

  Creates a summarizer to track the loss over time in TensorBoard.

  Creates an optimizer and applies the gradients to all trainable variables.

  The Op returned by this function is what must be passed to the
  `sess.run()` call to cause the model to train.

  Args:
    loss: Loss tensor, from loss().
    global_step: Integer Variable counting the number of training steps
      processed.
    learning_rate: The learning rate to use for gradient descent.

  Returns:
    train_op: The Op for training.
  """
  # Add a scalar summary for the snapshot loss.
  with tf.name_scope('train'):
    tf.scalar_summary(loss.op.name, loss)
    # Decay the learning rate exponentially based on the number of steps.
    # Variables that affect learning rate.


    rate = tf.train.exponential_decay(1.1*params.learning_rate,
                                      global_step,
                                      params.num_examples_per_epoch_for_train,
                                      0.9)
    tf.scalar_summary('learning_rate', rate)
    tf.scalar_summary('Global Step', global_step + 0.0)
    # Create the gradient descent optimizer with the given learning rate.
    optimizer = tf.train.AdamOptimizer(rate)
    # Use the optimizer to apply the gradients that minimize the loss
    # (and also increment the global step counter) as a single training step.
    train_op = optimizer.minimize(loss, global_step=global_step)
  return train_op