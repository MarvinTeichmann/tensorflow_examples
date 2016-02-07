import tensorflow as tf
import re

import data_input as data_input

# Global constants describing the CIFAR-10 data set.



IMAGE_SIZE = data_input.IMAGE_SIZE
NUM_CHANNELS = data_input.NUM_CHANNELS
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE
NUM_CLASSES = data_input.NUM_CLASSES
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = data_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = data_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL



flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('max_steps', 200000, 'Number of steps to run trainer.')
# flags.DEFINE_integer('num_filter_1', 32, 'Number of units in hidden layer 1.')
# flags.DEFINE_integer('num_filter_2', 64, 'Number of units in hidden layer 2.')


# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 350.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.1       # Initial learning rate.


def _variable_with_weight_decay(name, shape, stddev, wd):
  """Helper to create an initialized Variable with weight decay.

  Note that the Variable is initialized with a truncated normal distribution.
  A weight decay is added only if one is specified.

  Args:
    name: name of the variable
    shape: list of ints
    stddev: standard deviation of a truncated Gaussian
    wd: add L2Loss weight decay multiplied by this float. If None, weight
        decay is not added for this Variable.

  Returns:
    Variable Tensor
  """

  initial = tf.truncated_normal(shape, stddev=stddev)
  var = tf.Variable(initial, name=name)

  if wd:
    weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
  return var


def weight_variable(name, shape, stddev=0.1):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial, name=name)

def bias_variable(name, shape, constant=0.1):
  initial = tf.constant(constant, shape=shape)
  return tf.Variable(initial, name=name)
  
def conv2d(x, W, strides=[1, 1, 1, 1]):
  return tf.nn.conv2d(x, W, strides=strides, padding='SAME')

def max_pool_3x3(x, name):
  return tf.nn.max_pool(x, ksize=[1, 3, 3, 1],
                        strides=[1, 2, 2, 1], padding='SAME', name=name
                        )

def normalization(x,name):
  return tf.nn.lrn(x, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                    name=name)

def _activation_summary(x):
  """Helper to create summaries for activations.

  Creates a summary that provides a histogram of activations.
  Creates a summary that measure the sparsity of activations.

  Args:
    x: Tensor
  Returns:
    nothing
  """
  # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
  # session. This helps the clarity of presentation on tensorboard.
  tensor_name = x.op.name
  # tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
  tf.histogram_summary(tensor_name + '/activations', x)
  tf.scalar_summary(tensor_name + '/sparsity', tf.nn.zero_fraction(x))

def inference(images, keep_prob, train=True,
              num_filter_1=64, num_filter_2=64): 
  """Build the MNIST model up to where it may be used for inference.

  Args:
    images: Images placeholder, from inputs().
    num_filter_1: Amount of filters in conv1.
    num_filter_2: Amount of filters in conv2.

  Returns:
    softmax_linear: Output tensor with the computed logits.
  """

  # First Convolutional Layer
  with tf.variable_scope('Conv1') as scope:
    W_conv1 = _variable_with_weight_decay('weights', shape=[5, 5, NUM_CHANNELS, num_filter_1],
                                         stddev=1e-4, wd=0.0)
    b_conv1 = bias_variable('biases', [num_filter_1], constant=0.0)

    x_image = tf.reshape(images, [-1, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS])

    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1, name=scope.name)
    _activation_summary(h_conv1)


  # First Pooling Layer  
  h_pool1 = max_pool_3x3(h_conv1, name='pool1')
  # First Normalization
  norm1 = normalization(h_pool1, name='norm1')

  # Second Convolutional Layer
  with tf.name_scope('Conv2'):
    W_conv2 = _variable_with_weight_decay('weights', [5, 5, num_filter_1, num_filter_2],
                                          stddev=1e-4, wd=0.0)
    b_conv2 = bias_variable('biases', [num_filter_2])
    
    h_conv2 = tf.nn.relu(conv2d(norm1, W_conv2) + b_conv2)
    _activation_summary(h_conv2)

  # Second Pooling Layer
  h_pool2 = max_pool_3x3(h_conv2, name='pool2')

  # Second Normalization
  norm2 = normalization(h_pool2, name='norm1')

  # Fully Connected 1
  with tf.variable_scope('FullC1') as scope:
    # Move everything into depth so we can perform a single matrix multiply.
    dim = 1
    for d in norm2.get_shape()[1:].as_list():
      dim *= d
    reshape = tf.reshape(norm2, [FLAGS.batch_size, dim])

    weights = _variable_with_weight_decay('weights', shape=[dim, 384],
                                          stddev=0.04, wd=0.004)
    biases = bias_variable('biases', [384])
    fullc1 = tf.nn.relu_layer(reshape, weights, biases, name=scope.name)
    _activation_summary(fullc1)

  # local4
  with tf.variable_scope('FullC2') as scope:
    weights = _variable_with_weight_decay('weights', shape=[384, 192],
                                          stddev=0.04, wd=0.004)
    biases = bias_variable('biases', [192])
    fullc2 = tf.nn.relu_layer(fullc1, weights, biases, name=scope.name)
    _activation_summary(fullc2)



  # Computing Softmax
  with tf.name_scope('logits'):
    W_fc2 = _variable_with_weight_decay('weights', [192, NUM_CLASSES],
                                        stddev=1/192.0, wd=0.0)
    b_fc2 = bias_variable('biases', [NUM_CLASSES])
    logits = tf.matmul(fullc2, W_fc2) + b_fc2
    _activation_summary(logits)

  return logits


def loss(logits, labels):
  """Calculates the loss from the logits and the labels.

  Args:
    logits: Logits tensor, float - [batch_size, NUM_CLASSES].
    labels: Labels tensor, int32 - [batch_size].

  Returns:
    loss: Loss tensor of type float.
  """
  # Convert from sparse integer labels in the range [0, NUM_CLASSSES)
  # to 1-hot dense float vectors (that is we will have batch_size vectors,
  # each with NUM_CLASSES values, all of which are 0.0 except there will
  # be a 1.0 in the entry corresponding to the label).
  with tf.name_scope('loss'):
    batch_size = tf.size(labels)
    labels = tf.expand_dims(labels, 1)
    indices = tf.expand_dims(tf.range(0, batch_size), 1)
    concated = tf.concat(1, [indices, labels])
    onehot_labels = tf.sparse_to_dense(
        concated, tf.pack([batch_size, NUM_CLASSES]), 1.0, 0.0)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits,
                                                            onehot_labels,
                                                            name='xentropy')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='xentropy_mean')
    tf.add_to_collection('losses', cross_entropy_mean)

    loss = tf.add_n(tf.get_collection('losses'), name='total_loss')
  return loss

def _add_loss_summaries(total_loss):
  """Add summaries for losses in CIFAR-10 model.

  Generates moving average for all losses and associated summaries for
  visualizing the performance of the network.

  Args:
    total_loss: Total loss from loss().
  Returns:
    loss_averages_op: op for generating moving averages of losses.
  """
  # Compute the moving average of all individual losses and the total loss.
  loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
  losses = tf.get_collection('losses')
  loss_averages_op = loss_averages.apply(losses + [total_loss])

  # Attach a scalar summmary to all individual losses and the total loss; do the
  # same for the averaged version of the losses.
  for l in losses + [total_loss]:
    # Name each loss as '(raw)' and name the moving average version of the loss
    # as the original loss name.
    tf.scalar_summary(l.op.name +' (raw)', l)
    tf.scalar_summary(l.op.name, loss_averages.average(l))

  return loss_averages_op

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

  # Variables that affect learning rate.
  num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
  decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

  # Decay the learning rate exponentially based on the number of steps.
  lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                  global_step,
                                  decay_steps,
                                  LEARNING_RATE_DECAY_FACTOR,
                                  staircase=True)
  tf.scalar_summary('learning_rate', lr)

  # Generate moving averages of all losses and associated summaries.
  loss_averages_op = _add_loss_summaries(loss)

  # Compute gradients.
  with tf.control_dependencies([loss_averages_op]):
    opt = tf.train.GradientDescentOptimizer(lr)
    grads = opt.compute_gradients(loss)

  # Apply gradients.
  apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

  # Add histograms for trainable variables.
  for var in tf.trainable_variables():
    tf.histogram_summary(var.op.name, var)

  # Add histograms for gradients.
  for grad, var in grads:
    if grad:
      tf.histogram_summary(var.op.name + '/gradients', grad)

  # Track the moving averages of all trainable variables.
  variable_averages = tf.train.ExponentialMovingAverage(
      MOVING_AVERAGE_DECAY, global_step)
  variables_averages_op = variable_averages.apply(tf.trainable_variables())

  with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
    train_op = tf.no_op(name='train')

  return train_op

def evaluation(logits, labels):
  """Evaluate the quality of the logits at predicting the label.

  Args:
    logits: Logits tensor, float - [batch_size, NUM_CLASSES].
    labels: Labels tensor, int32 - [batch_size], with values in the
      range [0, NUM_CLASSES).

  Returns:
    A scalar int32 tensor with the number of examples (out of batch_size)
    that were predicted correctly.
  """
  # For a classifier model, we can use the in_top_k Op.
  # It returns a bool tensor with shape [batch_size] that is true for
  # the examples where the label's is was in the top k (here k=1)
  # of all logits for that example.
  with tf.name_scope('eval'):
    correct = tf.nn.in_top_k(logits, labels, 1)
    # Return the number of true entries.
    return tf.reduce_sum(tf.cast(correct, tf.int32))