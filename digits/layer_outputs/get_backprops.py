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


sys.path.insert(0,"/home/lzeerwanklyn/Projects/caffes/caffe-dvt/python")

# if complaining do this:
# sudo ldconfig /usr/local/cuda/lib64
os.environ['LD_LIBRARY_PATH'] = '/home/lzeerwanklyn/Projects/torches/torch-nv/install/lib:/home/lzeerwanklyn/Projects/nccl/build/lib:/usr/local/cuda-7.5/lib6'

import caffe

def write_deconv(image_path,deploy_prototxt, caffemodel,layer_name, neuron_index):
    delchars = ''.join(c for c in map(chr, range(256)) if not c.isalnum())

    # TODO: This should not be hardcoded!
    caffe.set_device(0)
    caffe.set_mode_gpu()

    # Load network
    net = caffe.Net(deploy_prototxt,caffemodel,caffe.TEST)

    # Setup data blob for input image
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2,0,1))
    transformer.set_raw_scale('data', 255)
    transformer.set_channel_swap('data', (2,1,0))

    # Load image and perform a forward pass
    image = caffe.io.load_image(image_path)
    transformed_image = transformer.preprocess('data', image)
    net.blobs['data'].data[...] = transformed_image
    net.forward()

    # Get deconvolution for specified layer and neuron
    diffs = net.blobs[layer_name].diff * 0


    diffs[0][int(neuron_index)] = net.blobs[layer_name].data[0,int(neuron_index)]
    net.backward_from_layer(layer_name, diffs, zero_higher = True)

    for key in net.blobs.keys():
        raw_data = net.blobs[key].diff[0]

        if raw_data.max() - raw_data.min() != 0.0:

            vis_data = utils.image.reshape_data_for_vis(raw_data,'BGR')[:10000]
            np.save(script_location+"/backprops/"+key.translate(None, delchars), utils.image.normalize_data(vis_data))



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Deconvolution tool - DIGITS')

    ### Positional arguments
    parser.add_argument('image_path',
            help='Input image (jpg,png,etc)')
    parser.add_argument('deploy_prototxt',
            help='File location of deploy.prototxt')
    parser.add_argument('caffemodel',
            help='File location of a **.caffemodel')
    parser.add_argument('layer_name',
            help='Name of layer (string)')
    parser.add_argument('neuron_index',
            help='neuron index from 0 to N')

    args = vars(parser.parse_args())
    write_deconv(
        args['image_path'],
        args['deploy_prototxt'],
        args['caffemodel'],
        args['layer_name'],
        args['neuron_index'],
            )
