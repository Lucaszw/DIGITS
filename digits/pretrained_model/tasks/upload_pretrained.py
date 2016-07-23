# Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.
from __future__ import absolute_import

import base64
from collections import OrderedDict
import h5py
import os.path
import tempfile
import re
import sys

import digits
from digits import device_query
from digits.task import Task
from digits.utils import subclass, override
from digits.status import Status
from digits import frameworks
from digits.framework_helpers import caffe_helpers

# TODO: Move to torch framework helpers
from digits.config import config_value


import subprocess

@subclass
class UploadPretrainedModelTask(Task):
    """
    A task for uploading pretrained models
    """

    def __init__(self, weights_path, model_def_path, image_info, labels_path=None, framework="caffe",**kwargs):
        """
        Arguments:
        weights_path -- path to model weights (**.caffemodel or ***.t7)
        model_def_path  -- path to model definition (**.prototxt or ***.lua)
        image_info -- a dictionary containing image_type, resize_mode, width, and height
        """
        self.weights_path = weights_path
        self.model_def_path = model_def_path
        self.image_info = image_info
        self.labels_path = labels_path
        self.framework = framework

        # resources
        self.gpu = None

        super(UploadPretrainedModelTask, self).__init__(**kwargs)

    @override
    def name(self):
        return 'Upload Pretrained Model'

    @override
    def __setstate__(self, state):
        super(UploadPretrainedModelTask, self).__setstate__(state)

    @override
    def process_output(self, line):
        return True

    @override
    def offer_resources(self, resources):
        reserved_resources = {}
        # we need one CPU resource from inference_task_pool
        cpu_key = 'inference_task_pool'
        if cpu_key not in resources:
            return None
        for resource in resources[cpu_key]:
            if resource.remaining() >= 1:
                reserved_resources[cpu_key] = [(resource.identifier, 1)]
                # we reserve the first available GPU, if there are any
                gpu_key = 'gpus'
                if resources[gpu_key]:
                    for resource in resources[gpu_key]:
                        if resource.remaining() >= 1:
                            self.gpu = int(resource.identifier)
                            reserved_resources[gpu_key] = [(resource.identifier, 1)]
                            break
                return reserved_resources
        return None

    def get_labels(self):
        labels = []
        if self.labels_path is not None:
            with open(self.job_dir+"/labels.txt") as f:
                labels = f.readlines()
        return labels

    def write_deploy(self,env):
        # get handle to framework object
        fw = frameworks.get_framework_by_id("caffe")
        model_def_path  = self.job_dir+"/original.prototxt"
        network = fw.get_network_from_path(model_def_path)

        channels  = int(self.image_info["image_type"])
        image_dim = [ int(self.image_info["width"]), int(self.image_info["height"]), channels ]

        caffe_helpers.save_deploy_file_classification(network,self.job_dir,len(self.get_labels()),None,image_dim,None)

    def write_torch_layers(self,env):
        # Write torch layers to json for layerwise graph visualization
        if config_value('torch_root') == '<PATHS>':
            torch_bin = 'th'
        else:
            torch_bin = os.path.join(config_value('torch_root'), 'bin', 'th')

        args = [torch_bin,
                os.path.join(os.path.dirname(os.path.dirname(digits.__file__)),'tools','torch','toJSON.lua'),
                '--network=%s' % "original",
                '--output=%s' % self.job_dir + "/model_def.json",
                ]

        p = subprocess.Popen(args,cwd=self.job_dir,env=env)

    def move_file(self,input, output,env):
        args  = ["cp", input, self.job_dir+"/"+output]
        p = subprocess.Popen(args,env=env)

    @override
    def run(self, resources):
        env = os.environ.copy()
        if self.framework == "caffe":
            self.move_file(self.weights_path, "model.caffemodel",env)
            self.move_file(self.model_def_path, "original.prototxt",env)
        else:
            self.move_file(self.weights_path, "_Model.t7",env)
            self.move_file(self.model_def_path, "original.lua",env)

        if self.labels_path is not None:
            self.move_file(self.labels_path, "labels.txt",env)

        if self.framework == "caffe":
            self.write_deploy(env)
        else:
            self.write_torch_layers(env)

        self.status = Status.DONE
