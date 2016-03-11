# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 11:50:47 2015

@author: teichman
"""

import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
from tensorflow.python.training import queue_runner
import os
import numpy as np


# Global constants describing out car data set.
IMAGE_SIZE = 32
NUM_CLASSES = 2

def input_pipeline(filename, batch_size, num_labels,
                   processing_image=lambda x:x,
                   processing_label=lambda y:y,
                   num_epochs=None):
                       
    """The input pipeline for reading images classification data.
     
    The data should be stored in a single text file of using the format:
     
     /path/to/image_0 label_0
     /path/to/image_1 label_1
     /path/to/image_2 label_2
     ...
    
     Args:
       filename: the path to the txt file
       batch_size: size of batches produced
       num_epochs: optionally limited the amount of epochs
      
    Returns:
       List with all filenames in file image_list_file
    """
    
    # Reads pfathes of images together with there labels
    image_list, label_list = read_labeled_image_list(filename)

                                                     
    images = ops.convert_to_tensor(image_list, dtype=dtypes.string)
    labels = ops.convert_to_tensor(label_list, dtype=dtypes.int32)
    
    # Makes an input queue
    input_queue = tf.train.slice_input_producer([images, labels],
                                                num_epochs=num_epochs,
                                                shuffle=True)

    # Reads the actual images from                                                 
    image, label = read_images_from_disk(input_queue,num_labels=num_labels)
    pr_image = processing_image(image)
    pr_label = processing_label(label)                             

    image_batch, label_batch = tf.train.batch([pr_image, pr_label],
                                              batch_size=batch_size)
    
    tf.image_summary('images', image_batch)                                                  
    return image_batch, label_batch


def inputs(filename, batch_size, num_labels,num_epochs=None):

  def pr_image(image):
    return tf.image.per_image_whitening(image)

  return input_pipeline(filename, batch_size,num_labels, processing_image=pr_image
                        ,num_epochs=None)

def distorted_inputs(filename, batch_size, num_labels,num_epochs=None):

  def pr_image(image):
    distorted_image = image
    # Randomly flip the image horizontally.
    distorted_image = tf.image.random_flip_left_right(distorted_image)

    # Because these operations are not commutative, consider randomizing
    # randomize the order their operation.
    distorted_image = tf.image.random_brightness(distorted_image,
                                                 max_delta=63)
    distorted_image = tf.image.random_contrast(distorted_image,
                                             lower=0.2, upper=1.8)

    return tf.image.per_image_whitening(distorted_image)

  return input_pipeline(filename, batch_size,num_labels, processing_image=pr_image
                        ,num_epochs=None)

        

def read_images_from_disk(input_queue, num_labels):
  """Consumes a single filename and label as a ' '-delimited string.

  Args:
    filename_and_label_tensor: A scalar string tensor.

  Returns:
    Two tensors: the decoded image, and the string label.
  """
  label = input_queue[1]
  file_contents = tf.read_file(input_queue[0])
  example = tf.image.decode_png(file_contents, channels=3)
  processed_example = preprocessing(example)
  # processed_labels = create_one_hot(label,num_labels)
  processed_label = label
  return processed_example, processed_label
  
  
def preprocessing(image):
    resized_image = tf.image.resize_images(image, IMAGE_SIZE,
                                           IMAGE_SIZE, method=0)
    resized_image.set_shape([IMAGE_SIZE,IMAGE_SIZE,3])
    return resized_image
    


def create_one_hot(label, num_labels = 10):
    """Produces one_hot vectors out of numerical labels
    
    Args:
       label_batch: a batch of labels
       num_labels: maximal number of labels
      
    Returns:
       Label Coded as one-hot vector
    """

    labels = tf.sparse_to_dense(label, [num_labels], 1.0, 0.0)
    
    return labels



def read_labeled_image_list(image_list_file):
    """Reads a .txt file containing pathes and labeles
    
    Args:
       image_list_file: a .txt file with one /path/to/image per line
       label: optionally, if set label will be pasted after each line
      
    Returns:
       List with all filenames in file image_list_file
    """
    f = open(image_list_file, 'r')
    filenames = []
    labels = []
    for line in f:
        filename, label = line[:-1].split(' ')
        filenames.append(filename)
        labels.append(int(label))
    return filenames, labels
        


        
def create_input_queues(image, label, capacity=100):
    """Creates Queues a FIFO Queue out of Input tensor objects.
     
     This function is no longer used in the input pipeline.
     However it took me a while to understand queuing and it might be useful
     fot someone at some point.

    Args:
       image: an image tensor object, generated by queues.
       label: an label tensor object, generated by queues.
      
    Returns: Two FiFO Queues
    """
    
    #create input queues

    im_queue = tf.FIFOQueue(capacity, dtypes.uint8)
    enqueue_op = im_queue.enqueue(image)
    
    queue_runner.add_queue_runner(queue_runner.QueueRunner(im_queue,
                                                           [enqueue_op]))

    label_queue = tf.FIFOQueue(capacity, dtypes.uint8)
    enqueue_op = label_queue.enqueue(label)
    
    queue_runner.add_queue_runner(queue_runner.QueueRunner(label_queue,
                                                           [enqueue_op]))
                                                           
    return im_queue, label_queue
    
def test_one_hot():
    data_folder = "/fzi/ids/teichman/no_backup/DATA/"
    data_file = "Vehicle_Data/train.txt"
    
    filename = os.path.join(data_folder, data_file)
    
        # Reads pfathes of images together with there labels
    image_list, label_list = read_labeled_image_list(filename)

                                                     
    images = ops.convert_to_tensor(image_list, dtype=dtypes.string)
    labels = ops.convert_to_tensor(label_list, dtype=dtypes.int32)
    
        # Makes an input queue
    input_queue = tf.train.slice_input_producer([images, labels],
                                                num_epochs=None,
                                                shuffle=True)

    # Reads the actual images from                                                 
    image, label = read_images_from_disk(input_queue, NUM_CLASSES)

    label_one_hot = create_one_hot(label,2)  
    
    init_op = tf.initialize_all_variables()
    sess = tf.Session()
    with sess.as_default():
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        
        print(sess.run([label,label_one_hot]))
        print(sess.run([label,label_one_hot]))
        print(sess.run([label,label_one_hot]))
        print(sess.run([label,label_one_hot]))
        print(sess.run([label,label_one_hot]))
        print(sess.run([label,label_one_hot]))
        

        sess.close()
    

    
def test_pipeline():
    data_folder = "/fzi/ids/teichman/no_backup/DATA/"
    data_file = "Vehicle_Data/test.txt"
    
    filename = os.path.join(data_folder, data_file)
    
    image_batch, label_batch = inputs(filename, 75,2)
    
    
    
    # Create the graph, etc.
    init_op = tf.initialize_all_variables()
    sess = tf.Session()
    with sess.as_default():
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        
        print(label_batch.eval())

        coord.request_stop()
        coord.join(threads)

        print("Finish Test")        
    
        sess.close()
 
    
if __name__ == '__main__':
  #test_one_hot()
  test_pipeline()
  test_pipeline()