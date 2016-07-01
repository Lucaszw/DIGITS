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

def get_predictions(labels_path,data):
    scores = data.flatten()
    indices = (-scores).argsort()
    with open(labels_path) as f:
        labels = f.readlines()

    top5 = []
    for i in range(0,5):
        top5.append({"label": labels[indices[i]], "score": scores[indices[i]].astype(float)})

    return top5

def get_stats(data, name, layer_type,index):
    mean = np.mean(data).astype(np.float32)
    std = np.std(data).astype(np.float32)
    y, x = np.histogram(data, bins=20)
    y = list(y.astype(float))
    ticks = x[[0,len(x)/2,-1]]
    x = [((x[i]+x[i+1])/2.0).astype(float) for i in xrange(len(x)-1)]
    ticks = list(ticks.astype(float))

    stats = {
        "shape": data.shape,
        "name": name,
        "type": layer_type,
        "mean": mean,
        "stddev": std,
        "histogram": json.dumps({"y": y, "x": x, "ticks": ticks }),
        "i": index
    }
    return stats

def get_weights(output_dir,net):
    # Write a database to hold vis information:
    f = h5py.File(output_dir+'/weights.hdf5','w')
    net.forward()
    # Save param keys to file:
    for index, key in enumerate(net.params.keys()):

        raw_data = net.params[key][0].data

        vis_data = utils.image.get_layer_vis_square(raw_data,
                allow_heatmap=bool(key != 'data'),
                channel_order = 'BGR')
        dset = f.create_dataset(key, data=utils.image.normalize_data(vis_data))

        stats = get_stats(raw_data,key,"Weights",index)
        dset.attrs.update(stats)

    f.close()

def get_activations(output_dir,net,input_list,labels_path):
    # TODO: retrive image dimensions and resize mode (height,width, channels)

    # load image paths from file
    image_paths = None
    with open(input_list) as infile:
        image_paths = infile.readlines()

    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2,0,1))
    transformer.set_raw_scale('data', 255)
    transformer.set_channel_swap('data', (2,1,0))

    f = h5py.File(output_dir+'/activations.hdf5','w')

    # load each image and perform a forward pass:
    for idx, path in enumerate(image_paths):
        path = path.strip()
        try:
            image = caffe.io.load_image(path)
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
                vis_data = utils.image.get_layer_vis_square(raw_data,
                        allow_heatmap=bool(key != 'data'),
                        channel_order = 'BGR')

                dset = grp.create_dataset(key, data=vis_data)

                stats = get_stats(raw_data,key,"Activations",index)
                dset.attrs.update(stats)

                if index + 1 == num_outputs:
                    if labels_path is not None:
                        predictions = get_predictions(labels_path,raw_data)
                        grp.attrs["predictions"] = json.dumps(predictions)
                    else:
                        print "NO LABELS!"
                        grp.attrs["predictions"] = ""


            logger.info('Processed %s/%s images',idx+1, len(image_paths))
        except utils.errors.LoadImageError as e:
            print e

    f.close()

def infer(input_list, output_dir, prototxt_path, caffemodel_path, labels_path, gpu):

    if gpu is not None:
        caffe.set_device(gpu)
        caffe.set_mode_gpu()
    else:
        caffe.set_mode_cpu()

    net = caffe.Net(prototxt_path,caffemodel_path,caffe.TEST)

    # TODO: Implement Mean Image File Option

    get_weights(output_dir, net)
    get_activations(output_dir, net, input_list, labels_path)

    logger.info('Saved data to %s', output_dir)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Inference tool for pretrained models - DIGITS')

    ### Positional arguments

    parser.add_argument('input_list',
            help='An input file containing paths to input data')

    parser.add_argument('output_dir',
            help='Directory to write outputs to')

    parser.add_argument('-p', '--prototxt_path',
            help='Path to deploy.prototxt',
            )

    parser.add_argument('-m', '--caffemodel_path',
            help='Path to caffemodel',
            )

    ### Optional arguments
    parser.add_argument('-l', '--labels_path',
            default=None,
            help='Path to labels.txt',
            )
    parser.add_argument('-g', '--gpu',
            type=int,
            default=None,
            help='GPU to use (as in nvidia-smi output, default: None)',
            )

    args = vars(parser.parse_args())

    try:
        infer(
            args['input_list'],
            args['output_dir'],
            args['prototxt_path'],
            args['caffemodel_path'],
            args['labels_path'],
            args['gpu']
            )
    except Exception as e:
        logger.error('%s: %s' % (type(e).__name__, e.message))
        raise
