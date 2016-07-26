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
import digits

from digits.config import load_config
load_config()

from digits import utils, log
from digits.inference.errors import InferenceError
from digits.framework_helpers import torch_helpers


logger = logging.getLogger('digits.tools.inference')

"""
Perform inference on a list of images using the specified model
"""

def get_activations(output_dir):

    activations_file = h5py.File(output_dir+'/activations.hdf5','a')
    vis_file = h5py.File(output_dir+'/vis.h5','r')

    # Create a group to hold image:
    index = str(len(activations_file.keys()))
    grp = activations_file.create_group(index)

    # Save param keys to file:
    num_outputs = len(vis_file['layers'].keys())
    for index, key in enumerate(vis_file['layers'].keys()):
        if 'activations' in vis_file['layers'][key]:
            chain = vis_file['layers'][key]['chain'][...].tostring()
            raw_data = vis_file['layers'][key]['activations'][...]

            if len(raw_data.shape)>1 and raw_data.shape[0]==1:
                raw_data = raw_data[0]

            vis_data = utils.image.reshape_data_for_vis(raw_data,'BGR')
            dset  = grp.create_dataset(chain, data=utils.image.normalize_data(vis_data))

            # TODO : Add stats here

        logger.info('Processed %s/%s layers', index, num_outputs)

    vis_file.close()
    activations_file.close()

def infer(image_path, model_def_path, weights_path, height, width, channels, resize_mode, labels_dir, gpu):

    if resize_mode is None:
        image_info = None
    else:
        image_info = {"height": height, "width": width, "channels": channels, "resize_mode": resize_mode}

    torch_helpers.save_activations_and_weights(
            image_path,
            model_def_path,
            weights_path,
            image_info,
            labels_dir,
            gpu,
            logger
            )

    get_activations(os.path.split(model_def_path)[0])

    logger.info('Saved data to %s', os.path.split(model_def_path)[0])

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
    parser.add_argument('-y', '--height',
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
    parser.add_argument('-l', '--labels_path',
            default=None,
            help='Path to labels text file',
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
            args['model_def_path'],
            args['weights_path'],
            args['height'],
            args['width'],
            args['channels'],
            args['resize_mode'],
            args['labels_path'],
            args['gpu']
            )
    except Exception as e:
        logger.error('%s: %s' % (type(e).__name__, e.message))
        raise
