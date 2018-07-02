# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import numpy as np
import h5py
from functools import partial
import matplotlib.pylab as plt
import tensorflow as tf

import sys
sys.path.append('../cnn/')
import model_large_promissing as sel_model

"""
Laplacian Pyramid Gradient Normalization
"""

def savearray(a, filename):
    a = np.uint8(np.clip(a, 0, 1)*255)
    plt.imshow(a)
    filename = filename.replace('/', '_')
    plt.savefig(filename+'.png')
    plt.close()
    np.save(filename+'.npy', a)

def visstd(a, s=0.1):
    """Normalize the image range for visualization"""
    return (a-a.mean())/max(a.std(), 1e-4)*s + 0.5

def lap_split(img):
    """Split the image into low and high frequency components."""
    k = np.float32([1, 4, 6, 4, 1])
    k = np.outer(k, k)
    k5x5 = np.expand_dims(np.expand_dims(k, -1), -1)
    #k5x5 = k[:, :, None, None] / k.sum() * np.eye(3, dtype=np.float32)
    with tf.name_scope('split'):
        lo = tf.nn.conv2d(img, k5x5, [1, 2, 2, 1], 'SAME')
        lo2 = tf.nn.conv2d_transpose(lo, k5x5*4, tf.shape(img), [1, 2, 2, 1])
        hi = img - lo2
    return lo, hi

def lap_split_n(img, n):
    """Build Laplacian pyramid with n splits."""
    levels = []
    for i in range(n):
        img, hi = lap_split(img)
        levels.append(hi)
    levels.append(img)
    return levels[::-1]

def lap_merge(levels):
    """Merge Laplacian pyramid."""
    img = levels[0]
    k = np.float32([1, 4, 6, 4, 1])
    k = np.outer(k, k)
    k5x5 = np.expand_dims(np.expand_dims(k, -1), -1)
    #k5x5 = k[:, :, None, None] / k.sum() * np.eye(3, dtype=np.float32)
    for hi in levels[1:]:
        with tf.name_scope('merge'):
            img = tf.nn.conv2d_transpose(img, k5x5*4, tf.shape(hi),
                                         [1, 2, 2, 1]) + hi
    return img

def normalize_std(img, eps=1e-10):
    """Normalize image by making its standard deviation=1.0"""
    with tf.name_scope('normalize'):
        std = tf.sqrt(tf.reduce_mean(tf.square(img)))
        return img/tf.maximum(std, eps)

def lap_normalize(img, scale_n=4):
    """Perform the Laplacian pyramid normalization."""
    img = tf.expand_dims(tf.expand_dims(img, -1), 0)
    tlevels = lap_split_n(img, scale_n)
    tlevels = list(map(normalize_std, tlevels))
    out = lap_merge(tlevels)
    print out.shape
    return out[0, :, :, :]

def resize(img, size):
    """Helper function that uses TF to resize an image"""
    img = tf.expand_dims(tf.expand_dims(img, 0), -1)
    return tf.image.resize_bilinear(img, size)[0, :, :, 0]


if __name__=='__main__':
    base_dir = r'/nfs/home/huanglijie/repo/Facial-Recognition'
    model_dir = os.path.join(base_dir,'exp','cnn','log_model_large_promissing_')
    model_data = os.path.join(model_dir, 'checkpoint_49.ckpt')
    dataset_dir = os.path.join(base_dir, 'data_hdf5')

    # load preprocessing parameters for input
    input_scale_file = os.path.join(dataset_dir,'training_images_mean_scale.h5')
    f = h5py.File(input_scale_file)
    image_scale = f['scale'][()]
    image_mean = f['mean'][:]

    # load the model
    is_training = False
    with tf.device('/gpu:0'):
        t_input = tf.placeholder(tf.float32, shape=(1, 48, 48))
        t_preprocessed = (t_input - image_mean) * image_scale
        is_training_ph = tf.placeholder(tf.bool, shape=())
        net = sel_model.get_model(t_preprocessed, is_training=is_training_ph,
                                  cat_num=7, weight_decay=0.0, bn_decay=0.0)
    saver = tf.train.Saver()
    # create tensorflow session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    sess = tf.Session(config=config)
    saver.restore(sess, model_data)
    graph = sess.graph

    # get Conv2D layer name
    layers = [op.name for op in graph.get_operations() if op.type=='Conv2D']
    feature_nums = [int(graph.get_tensor_by_name(name+':0').get_shape()[-1]) 
                        for name in layers]
    print('Number of layers', len(layers))
    print('Total number of feature channels:', sum(feature_nums))

    # Helper function
    def tffunc(*argtypes):
        """Helper that transforms TF-graph generating function into a regular
        one.
        """
        placeholders = list(map(tf.placeholder, argtypes))
        def wrap(f):
            out = f(*placeholders)
            def wrapper(*args, **kw):
                return out.eval(dict(zip(placeholders, args)), session=sess)
            return wrapper
        return wrap

    def calc_grad_tiled(img, t_grad, tile_size=48):
        """Compute the value of tensor t_grad over the image in a tiled way.
        Random shifts are applied to the image to blur tile  boundaries over
        multiple iterations.
        """
        sz = tile_size
        h, w = img.shape[:2]
        sx, sy = np.random.randint(sz, size=2)
        img_shift = np.roll(np.roll(img, sx, 1), sy, 0)
        grad = np.zeros_like(img)
        for y in range(0, max(h-sz//2, sz), sz):
            for x in range(0, max(w-sz//2, sz), sz):
                sub = img_shift[y:y+sz, x:x+sz]
                g = sess.run(t_grad, {t_input: np.expand_dims(sub, 0),
                                      is_training_ph: is_training})
                grad[y:y+sz, x:x+sz] = g[0, :, :]
        return np.roll(np.roll(grad, -sx, 1), -sy, 0)
    
    resize = tffunc(np.float32, np.int32)(resize)

    # build the laplacian normalization graph
    lap_norm_func = tffunc(np.float32)(partial(lap_normalize, scale_n=4))
    
    for layer in layers:
        channel_num = int(graph.get_tensor_by_name(layer+':0').get_shape()[-1])
        for channel in range(channel_num):
            # start with a gray image with a little noise
            img_noise = np.random.uniform(size=(48, 48)) + 115.0
            print 'Viz feature of Layer %s, Channel %s'%(layer, channel)
            t_obj = graph.get_tensor_by_name('%s:0'%layer)[:, :, :, channel]
 
            # defining the optimization objective
            t_score = tf.reduce_mean(t_obj)
            # behold the power of automatic differentiation!
            t_grad = tf.gradients(t_score, t_input)[0]

            img = img_noise.copy()
            for octave in range(3):
                if octave>0:
                    hw = np.float32(img.shape[:2]) * 1.5
                    img = resize(img, np.int32(hw))
                for i in range(10):
                    g = calc_grad_tiled(img, t_grad)
                    g = lap_norm_func(g)
                    img += g[:, :, 0]*1.0
                    print '.',
            savearray(visstd(img), '%s_%s'%(layer, channel))
 
