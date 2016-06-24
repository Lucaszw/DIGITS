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


def get_weights(prototxt,caffemodel,output_path):
    caffe.set_device(0)
    caffe.set_mode_gpu()

    net = caffe.Net(prototxt,caffemodel,caffe.TEST)
    net.forward()

    # Write a database to hold vis information:
    f = h5py.File(output_path+'/weights.hdf5','w-')

    # Save param keys to file:
    for key in net.params.keys():
        raw_data = net.params[key][0].data
        vis_data = utils.image.reshape_data_for_vis(raw_data,'BGR')[:10000]
        dset = f.create_dataset(key, data=utils.image.normalize_data(vis_data))

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Save Weights of Caffe Model to DB')

    parser.add_argument('prototxt',
            help='deploy.prototxt file location')

    parser.add_argument('caffemodel',
            help='***.caffemodel file location')

    parser.add_argument('output_path',
            help='folder to write outputs to')

    args = vars(parser.parse_args())

    try:
        get_weights(
            args['prototxt'],
            args['caffemodel'],
            args['output_path']
            )
    except Exception as e:
        print e.message
        raise
