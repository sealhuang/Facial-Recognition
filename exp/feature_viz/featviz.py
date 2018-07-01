# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import sys
import numpy as np
import PIL.Image
from functools import partial

import tensorflow as tf

base_dir = r'/nfs/home/huanglijie/repo/Facial-Recognition/exp/cnn'
model_dir = os.path.join(base_dir, 'log_model_large_promissing_')
model_meta = os.path.join(model_dir, 'checkpoint_49.ckpt.meta')
model_data = os.path.join(base_dir, 'checkpoint_49.ckpt')

# creating TensorFlow session and loading the model
graph = tf.Graph()
sess = tf.Session(graph=graph)
saver = tf.train.import_meta_graph(model_meta)
saver.restore(sess, model_data)
