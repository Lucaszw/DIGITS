import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import h5py

script_location = os.path.dirname(os.path.realpath(sys.argv[0]))
sys.path.insert(0,script_location+"/../..")

import digits
from digits import utils
import argparse
from subprocess import call

sys.path.insert(0,"/home/lzeerwanklyn/Projects/caffes/caffe-nv/python")
import caffe


def get_activations(image_path,prototxt,caffemodel,output_path):
    caffe.set_device(0)
    caffe.set_mode_gpu()

    net = caffe.Net(prototxt,caffemodel,caffe.TEST)
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2,0,1))
    transformer.set_raw_scale('data', 255)
    transformer.set_channel_swap('data', (2,1,0))

    image = caffe.io.load_image(image_path)
    transformed_image = transformer.preprocess('data', image)
    net.blobs['data'].data[...] = transformed_image
    net.forward()

    # Create a group to hold image:
    f = h5py.File(output_path+'/activations.hdf5','a')
    group_name = str(len(f.keys()))
    grp = f.create_group(group_name)

    # Save param keys to file:
    for key in net.blobs.keys():
        raw_data = net.blobs[key].data[0]
        vis_data = utils.image.reshape_data_for_vis(raw_data,'BGR')[:10000]
        grp.create_dataset(key, data=utils.image.normalize_data(vis_data))

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Save Weights of Caffe Model to DB')

    parser.add_argument('image_path',
            help='path to input image (.jpg, .png, ...)')

    parser.add_argument('prototxt',
            help='deploy.prototxt file location')

    parser.add_argument('caffemodel',
            help='***.caffemodel file location')

    parser.add_argument('output_path',
            help='folder to write outputs to')

    args = vars(parser.parse_args())

    try:
        get_activations(
            args['image_path'],
            args['prototxt'],
            args['caffemodel'],
            args['output_path']
            )
    except Exception as e:
        print e.message
        raise
