# -*- coding: utf-8 -*-
"""
Created on Sat Feb  6 19:24:19 2016

@author: teichman
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import time
import numpy as np

import gzip
import os
import re
import sys
import zipfile
import random
import logging
import scipy as scp
import scipy.misc
from six.moves import urllib


#import tensorflow as tf

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.DEBUG,
                    stream=sys.stdout)


STRIDE = 10
IMAGE_SIZE = 50
NUM_PIXELS = IMAGE_SIZE*IMAGE_SIZE
DATA_URL = "http://kitti.is.tue.mpg.de/kitti/data_road.zip"

#flags = tf.app.flags
#FLAGS = flags.FLAGS
#flags.DEFINE_string('data_dir', 'data',
#                    'Directory to put the training data.')
#flags.DEFINE_integer('batch_size', 100, 'Batch size.  '
#                     'Must divide evenly into the dataset sizes.')




def maybe_download_and_extract():
  """Download and extract the tarball from Alex's website."""
  flags = tf.app.flags
  FLAGS = flags.FLAGS
  dest_directory = FLAGS.data_dir
  if not os.path.exists(dest_directory):
    os.makedirs(dest_directory)
  filename = DATA_URL.split('/')[-1]
  filepath = os.path.join(dest_directory, filename)
  if not os.path.exists(filepath):
    def _progress(count, block_size, total_size):
      sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename,
          float(count * block_size) / float(total_size) * 100.0))
      sys.stdout.flush()
    filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath,
                                             reporthook=_progress)
    print()
    statinfo = os.stat(filepath)
    print('Succesfully downloaded', filename, statinfo.st_size, 'bytes.')
    zipfile.ZipFile(filepath, 'r').extractall(dest_directory)
    process_data(dest_directory)

def process_data(dest_directory):
  # this are the dictionaries of data    
  path_data = os.path.join(dest_directory,"data_road/training/image_2/")
  path_gt = os.path.join(dest_directory,"data_road/training/gt_image_2/")

  # Load lists of files and names, random shuffel them
  names = [f for f in sorted(os.listdir(path_data)) if f.endswith('.png')]
  random.shuffle(names)
  num_names = len(names)

  road_snippets = os.path.join(dest_directory, "road")
  os.makedirs(road_snippets)
  bg_snippets = os.path.join(dest_directory, "background")
  os.makedirs(bg_snippets)

  train_file_name = os.path.join(dest_directory, "train.txt")
  test_file_name = os.path.join(dest_directory, "test.txt")

  train_file = open(train_file_name, "w")
  test_file = open(test_file_name, "w")

  for i, image_file in enumerate(names):

    # Copy Names of Images
    logging.info("Processing Image %i / %i : %s", i,num_names, image_file)
    data_file = os.path.join(path_data, image_file)
    gt_name = image_file.split('_')[0] + "_road_" + image_file.split('_')[1]
    gt_file = os.path.join(path_gt, gt_name)

    data = scp.misc.imread(data_file)
    gt = scp.misc.imread(gt_file)

    # mygt == 0 iff pixel is background
    mygt = 255!=np.sum(gt, axis=2)

    skip = 0      
    for x in range(mygt.shape[0]-IMAGE_SIZE, 0 , -STRIDE):
      for y in range(0, mygt.shape[1]-IMAGE_SIZE, STRIDE):
          if (skip>0):
            skip = skip-1
            continue
          if(np.sum(mygt[x:(x+IMAGE_SIZE),y:(y+IMAGE_SIZE)])==0):
              file_name = "%s_%i_%i.png" % (image_file.split('.')[0], x, y)
              save_file = os.path.join(bg_snippets, file_name)
              scp.misc.imsave(save_file, data[x:(x+IMAGE_SIZE),y:(y+IMAGE_SIZE)])
              skip = skip + 2
              if i < 200:
                train_file.write(save_file + " 0" + "\n")
              else:
                test_file.write(save_file + " 0" + "\n")
          elif(np.sum(mygt[x:(x+IMAGE_SIZE),y:(y+IMAGE_SIZE)]) > 0.8*NUM_PIXELS):
              file_name = "%s_%i_%i.png" % (image_file.split('.')[0], x, y)
              save_file = os.path.join(road_snippets, file_name)
              scp.misc.imsave(save_file, data[x:(x+IMAGE_SIZE),y:(y+IMAGE_SIZE)])
              if i < 200:
                train_file.write(save_file + " 1" + "\n")
              else:
                test_file.write(save_file + " 1" + "\n")

if __name__ == '__main__':
  process_data('data')




                
              