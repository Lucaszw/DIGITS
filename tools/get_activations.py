#!/usr/bin/env python2
# Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.

import argparse
import h5py
import json
import logging
import numpy as np
import os
import sys

# Add path for DIGITS package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import digits.config
digits.config.load_config()

from digits import utils, log
from digits.inference.errors import InferenceError

# must call digits.config.load_config() before caffe to set the path
import caffe

logger = logging.getLogger('digits.tools.inference')

"""
Perform inference on a list of images using the specified model
"""

def get_activations(output_dir,net,image_path, height, width, channels, resize_mode):
    # TODO: retrive image dimensions and resize mode (height,width, channels)

    image = utils.image.load_image(image_path)
    image = utils.image.resize_image(image,
                height, width,
                channels    = channels,
                resize_mode = resize_mode,
                )

    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2,0,1))

    # TODO: Implement Mean Image File Option

    # transformer.set_raw_scale('data', 255)

    if channels == 3:
        # BGR when there are three channels
        # XXX see issue #59
        channel_swap = (2,1,0)
        transformer.set_channel_swap('data', (2,1,0))

    f = h5py.File(output_dir+'/activations.hdf5','a')

    # load the image and perform a forward pass:
    try:
        image = caffe.io.load_image(image_path)
        transformed_image = transformer.preprocess('data', image)
        net.blobs['data'].data[...] = transformed_image
        net.forward()

        # Create a group to hold image:
        index = str(len(f.keys()))
        grp = f.create_group(index)

        # Save param keys to file:
        num_outputs = len(net.blobs.keys())
        for index, key in enumerate(net.blobs.keys()):
            raw_data = net.blobs[key].data[0]

            vis_data = utils.image.reshape_data_for_vis(raw_data,'BGR')[:10000]
            dset = grp.create_dataset(key, data=utils.image.normalize_data(vis_data))
            # TODO - Add stats here
            logger.info('Processed %s/%s blobs', index, num_outputs)

    except utils.errors.LoadImageError as e:
        print e

    f.close()

def infer(image_path, output_dir, model_def_path, weights_path, height, width, channels, resize_mode, gpu):

    if gpu is not None:
        caffe.set_device(gpu)
        caffe.set_mode_gpu()
    else:
        caffe.set_mode_cpu()

    net = caffe.Net(model_def_path,weights_path,caffe.TEST)

    get_activations(output_dir, net, image_path, height, width, channels, resize_mode)

    logger.info('Saved data to %s', output_dir)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Activations tool for pretrained models - DIGITS')

    ### Positional arguments

    parser.add_argument('image_path',
            help='Path to input image')

    parser.add_argument('output_dir',
            help='Directory to write outputs to')

    parser.add_argument('-p', '--model_def_path',
            help='Path to model definition',
            )

    parser.add_argument('-m', '--weights_path',
            help='Path to weights',
            )

    ### Optional arguments
    parser.add_argument('-l', '--height',
            type=int,
            default=256,
            help='Image Height (ex. 224)',
            )
    parser.add_argument('-w', '--width',
            type=int,
            default=256,
            help='Image Width (ex. 224)',
            )
    parser.add_argument('-c', '--channels',
            type=int,
            default=3,
            help='Number of Channels (ex. 3 for RGB, 1 for Grayscale)',
            )
    parser.add_argument('-r', '--resize_mode',
            default="squash",
            help='Resize Mode (ex. squash, crop, fill, half_crop)',
            )
    parser.add_argument('-g', '--gpu',
            type=int,
            default=None,
            help='GPU to use (as in nvidia-smi output, default: None)',
            )

    args = vars(parser.parse_args())

    try:
        infer(
            args['image_path'],
            args['output_dir'],
            args['model_def_path'],
            args['weights_path'],
            args['height'],
            args['width'],
            args['channels'],
            args['resize_mode'],
            args['gpu']
            )
    except Exception as e:
        logger.error('%s: %s' % (type(e).__name__, e.message))
        raise
