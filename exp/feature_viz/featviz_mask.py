# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import numpy as np
#from scipy.misc import imresize
import h5py
import matplotlib.pylab as plt
import tensorflow as tf

import sys
sys.path.append('../cnn/')
import model_large_promissing as sel_model


def savearray(img, mask, filename):
    mask[mask==0] = 0.5
    nimg = img * mask
    plt.imshow(nimg, cmap='gray', vmin=0, vmax=255)
    plt.savefig(filename)
    plt.close()
    #np.save(filename+'.npy', mask)

def visstd(a, s=0.1):
    """Normalize the image range for visualization"""
    return (a-a.mean())/max(a.std(), 1e-4)*s + 0.5

def load_h5(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    return (data, label)

def load_h5_mean_scale(h5_filename):
    f = h5py.File(h5_filename)
    image_scale = f['scale'][()]
    image_mean = f['mean'][:]
    return (image_mean, image_scale)


if __name__=='__main__':
    base_dir = r'/nfs/home/huanglijie/repo/Facial-Recognition'
    model_dir = os.path.join(base_dir,'exp','cnn','log_model_large_promissing_')
    model_data = os.path.join(model_dir, 'checkpoint_49.ckpt')
    dataset_dir = os.path.join(base_dir, 'data_hdf5')
    feat_dir = os.path.join(base_dir, 'exp', 'feature_viz', 'features')

    # load preprocessing parameters for input
    input_scale_file = os.path.join(dataset_dir,'training_images_mean_scale.h5')
    image_mean, image_scale = load_h5_mean_scale(input_scale_file)

    # load testing images
    testing_file = os.path.join(dataset_dir, 'test_0.h5')
    imgs, labels = load_h5(testing_file)

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

    for i in range(30):
        # start with a selected image from testing images
        print 'Image %s'%(i*2+1)
        img0 = imgs[i*2+1]
        for layer in layers:
            channel_num = int(graph.get_tensor_by_name(layer+':0').get_shape()[-1])
            print 'Viz feature of Layer %s'%(layer)
            t_obj = graph.get_tensor_by_name('%s:0'%layer)
            t_obj = tf.image.resize_bilinear(t_obj, [48, 48])[0, :, :, :]

            img = np.float32(img0.copy())
            t = sess.run([t_obj], {t_input: np.expand_dims(img, 0),
                                   is_training_ph: is_training})
            t = t[0]
            for channel in range(channel_num):
                tmp_t = t[:, :, channel]
                tmp_t = (tmp_t - tmp_t.min()) / (tmp_t.max() - tmp_t.min())
                tmp_t[tmp_t<0.6] = 0
                tmp_t[tmp_t>0] = 1
                l = layer.replace('/', '_')
                out_dir = os.path.join(feat_dir, '%s_%s'%(l, channel))
                if not os.path.exists(out_dir):
                    os.makedirs(out_dir)
                savearray(img0, tmp_t,os.path.join(out_dir,'img%s.png'%(i*2+1)))


