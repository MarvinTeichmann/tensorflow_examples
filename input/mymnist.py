import input_data
import tensorflow as tf
import re

IMAGE_SIZE = 28
NUM_CHANNELS = 1
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE
NUM_CLASSES = 10
TOWER_NAME = 'tower'

def weight_variable(name, shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial, name=name)

def bias_variable(name, shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial, name=name)
  
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x, name):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME', name=name
                        )

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
  tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
  tf.histogram_summary(tensor_name + '/activations', x)
  tf.scalar_summary(tensor_name + '/sparsity', tf.nn.zero_fraction(x))

def inference(images, keep_prob, train=True, num_filter_1=32, num_filter_2=64): 
  """Build the MNIST model up to where it may be used for inference.

  Args:
    images: Images placeholder, from inputs().
    num_filter_1: Amount of filters in conv1.
    num_filter_2: Amount of filters in conv2.

  Returns:
    softmax_linear: Output tensor with the computed logits.
  """

  # First Convolutional Layer
  with tf.name_scope('Conv1'):
    # Adding Convolutional Layers                        
    W_conv1 = weight_variable('weights', [5, 5, NUM_CHANNELS, num_filter_1])
    b_conv1 = bias_variable('biases', [num_filter_1])
      
    x_image = tf.reshape(images, [-1, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS])
    
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    _activation_summary(h_conv1)
    
  # First Pooling Layer  
  h_pool1 = max_pool_2x2(h_conv1, name='pool1')

  # Second Convolutional Layer
  with tf.name_scope('Conv2'):
    W_conv2 = weight_variable('weights', [5, 5, num_filter_1, num_filter_2])
    b_conv2 = bias_variable('biases', [num_filter_2])
    
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    _activation_summary(h_conv2)

  # Second Pooling Layer
  h_pool2 = max_pool_2x2(h_conv2, name='pool2')


  # Adding Fully Connected Layers
  with tf.name_scope('fc1'):
    W_fc1 = weight_variable('weights', [7 * 7 * num_filter_2, 1024])
    b_fc1 = bias_variable('biases',[1024])
    
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*num_filter_2])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    _activation_summary(h_fc1)

  # Adding Dropout
  h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob, name='dropout')

  with tf.name_scope('logits'):
    W_fc2 = weight_variable('weights', [1024, 10])
    b_fc2 = bias_variable('biases', [10])
    logits = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
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
    loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')
  return loss

def training(loss, learning_rate):
  """Sets up the training Ops.

  Creates a summarizer to track the loss over time in TensorBoard.

  Creates an optimizer and applies the gradients to all trainable variables.

  The Op returned by this function is what must be passed to the
  `sess.run()` call to cause the model to train.

  Args:
    loss: Loss tensor, from loss().
    learning_rate: The learning rate to use for gradient descent.

  Returns:
    train_op: The Op for training.
  """
  # Add a scalar summary for the snapshot loss.
  with tf.name_scope('train'):
    tf.scalar_summary(loss.op.name, loss)
    # Create the gradient descent optimizer with the given learning rate.
    optimizer = tf.train.AdamOptimizer(1e-4)
    # optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    # Create a variable to track the global step.
    global_step = tf.Variable(0, name='global_step', trainable=False)
    # Use the optimizer to apply the gradients that minimize the loss
    # (and also increment the global step counter) as a single training step.
    train_op = optimizer.minimize(loss, global_step=global_step)
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









