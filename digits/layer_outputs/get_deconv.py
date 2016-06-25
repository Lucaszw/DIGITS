import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import argparse

script_location = os.path.dirname(os.path.realpath(sys.argv[0]))
sys.path.insert(0,script_location+"/../..")

import digits
from digits import utils
import argparse
from subprocess import call
import h5py


sys.path.insert(0,"/home/lzeerwanklyn/Projects/caffes/caffe-dvt/python")

# if complaining do this:
# sudo ldconfig /usr/local/cuda/lib64
os.environ['LD_LIBRARY_PATH'] = '/home/lzeerwanklyn/Projects/torches/torch-nv/install/lib:/home/lzeerwanklyn/Projects/nccl/build/lib:/usr/local/cuda-7.5/lib6'

import caffe

def write_deconv(job_path,image_key,layer_name, neuron_index):
    delchars = ''.join(c for c in map(chr, range(256)) if not c.isalnum())

    # TODO: This should not be hardcoded!
    caffe.set_device(0)
    caffe.set_mode_gpu()

    deploy_prototxt = job_path+"/deploy.prototxt"
    caffemodel = job_path+"/model.caffemodel"

    # Load network
    net = caffe.Net(deploy_prototxt,caffemodel,caffe.TEST)

    # Setup data blob for input image
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2,0,1))
    transformer.set_raw_scale('data', 255)
    transformer.set_channel_swap('data', (2,1,0))

    # Load image and perform a forward pass
    f = h5py.File(job_path+'/activations.hdf5','r')
    image = f[image_key]['data'][:][0]

    transformed_image = transformer.preprocess('data', image)
    net.blobs['data'].data[...] = transformed_image
    net.forward()

    # Get deconvolution for specified layer and neuron
    diffs = net.blobs[layer_name].diff * 0


    diffs[0][int(neuron_index)] = net.blobs[layer_name].data[0,int(neuron_index)]
    net.deconv_from_layer(layer_name, diffs, zero_higher = True)

    # Save data to numpy array
    raw_data = net.blobs['data'].diff[0]
    vis_data = utils.image.reshape_data_for_vis(raw_data,'BGR')[:10000]

    np.save(script_location+"/deconv/"+layer_name.translate(None, delchars), utils.image.normalize_data(vis_data))



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Deconvolution tool - DIGITS')

    ### Positional arguments
    parser.add_argument('job_path',
            help='Path to job containing deploy.prototxt and caffemodel')
    parser.add_argument('image_key',
            help='Group key of dataset containing image in activations.hdf5')
    parser.add_argument('layer_name',
            help='Name of layer (string)')
    parser.add_argument('neuron_index',
            help='neuron index from 0 to N')

    args = vars(parser.parse_args())
    write_deconv(
        args['job_path'],
        args['image_key'],
        args['layer_name'],
        args['neuron_index'],
            )
