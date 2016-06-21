import sys
import os
import numpy as np
import matplotlib.pyplot as plt

script_location = os.path.dirname(os.path.realpath(sys.argv[0]))
sys.path.insert(0,script_location+"/../..")

import digits
from digits import utils
import argparse
from subprocess import call

sys.path.insert(0,"/home/lzeerwanklyn/Projects/caffes/caffe-nv/python")
import caffe


def generate_visualizations(image_path,prototxt,caffemodel):
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
    delchars = ''.join(c for c in map(chr, range(256)) if not c.isalnum())

    call(["rm","-r",script_location+"/blobs/*"])
    call(["rm","-r",script_location+"/params/*"])

    # Save param keys to file:
    np.save(script_location+"/param_keys",net.params.keys())
    for key in net.params.keys():
        raw_data = net.params[key][0].data
        vis_data = utils.image.reshape_data_for_vis(raw_data,'BGR')[:10000]
        np.save(script_location+"/params/"+key.translate(None, delchars), utils.image.normalize_data(vis_data))

    np.save(script_location+"/blob_keys",net.blobs.keys())
    for key in net.blobs.keys():
        raw_data = net.blobs[key].data[0]
        vis_data = utils.image.reshape_data_for_vis(raw_data,'BGR')[:10000]
        np.save(script_location+"/blobs/"+key.translate(None, delchars), utils.image.normalize_data(vis_data))



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Vis Data Generator')

    ### Positional arguments
    parser.add_argument('image',
            help='**.jpg, ***.png; Input image file location')

    parser.add_argument('prototxt',
            help='deploy.prototxt file location')

    parser.add_argument('caffemodel',
            help='***.caffemodel file location')

    args = vars(parser.parse_args())

    try:
        generate_visualizations(
            args['image'],
            args['prototxt'],
            args['caffemodel'],
            )
    except Exception as e:
        print e.message
        raise
