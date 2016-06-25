import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import argparse
import h5py
script_location = os.path.dirname(os.path.realpath(sys.argv[0]))
sys.path.insert(0,script_location+"/../..")

import digits
from digits import utils
import argparse
from subprocess import call

sys.path.insert(0,"/home/lzeerwanklyn/Projects/caffes/caffe-dvt/python")

# if complaining do this:
# sudo ldconfig /usr/local/cuda/lib64
os.environ['LD_LIBRARY_PATH'] = '/home/lzeerwanklyn/Projects/torches/torch-nv/install/lib:/home/lzeerwanklyn/Projects/nccl/build/lib:/usr/local/cuda-7.5/lib6'

import caffe

def write_backprops(job_path,layer_name,neuron_index):
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

    activations = h5py.File(job_path+'/activations.hdf5','r')
    f = h5py.File(job_path+'/backprops.hdf5','w')

    # Load each image and perform a forward pass
    for image_key in activations:
        image = activations[image_key]['data'][:][0]

        # Perform forward pass on net:
        transformed_image = transformer.preprocess('data', image)
        net.blobs['data'].data[...] = transformed_image
        net.forward()

        # Get backprops for specified layer and neuron
        diffs = net.blobs[layer_name].diff * 0
        diffs[0][int(neuron_index)] = net.blobs[layer_name].data[0,int(neuron_index)]
        net.backward_from_layer(layer_name, diffs, zero_higher = True)

        # Create a group to hold image:
        grp  = f.create_group(image_key)
        info = grp.create_dataset("info", data=activations[image_key][layer_name][int(neuron_index)])
        info.attrs['layer']  = layer_name
        info.attrs['neuron'] = neuron_index

        # Save all the backprop data for the layers:
        for key in net.blobs.keys():
            raw_data = net.blobs[key].diff[0]

            # Dont store blobs with no data (layers above backprop operation):
            if raw_data.max() - raw_data.min() != 0.0:
                vis_data = utils.image.reshape_data_for_vis(raw_data,'BGR')[:10000]
                grp.create_dataset(key, data=utils.image.normalize_data(vis_data))

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Backprops tool - DIGITS')

    ### Positional arguments
    parser.add_argument('job_path',
            help='Path to job containing deploy.prototxt, caffemodel and activations.h5py')
    parser.add_argument('layer_name',
            help='Name of layer (string)')
    parser.add_argument('neuron_index',
            help='neuron index from 0 to N')

    args = vars(parser.parse_args())
    write_backprops(
        args['job_path'],
        args['layer_name'],
        args['neuron_index'],
            )
