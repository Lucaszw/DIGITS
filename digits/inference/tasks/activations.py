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
from digits.utils.image import embed_image_html
from digits.status import Status
from digits.inference.errors import InferenceError
# from digits.pretrained_model import PretrainedModelJob

@subclass
class ActivationsTask(Task):
    """
    Get Activations from a Pretrained Model
    """

    def __init__(self, pretrained_model, image_path, **kwargs):
        """
        Arguments:
        pretrained_model -- job relating to a pretrained model
        image_path -- image to perform inference on
        """

        # memorize parameters
        self.image_path = image_path

        self.weights_path = pretrained_model.get_weights_path()
        self.deploy_path = pretrained_model.get_deploy_path()
        self.image_info = pretrained_model.image_info

        self.pretrained_model = pretrained_model
        self.inference_log_file = "inference.log"

        # resources
        self.gpu = None

        # generated data
        self.inference_data_filename = None
        self.inference_inputs = None
        self.inference_outputs = None
        self.inference_layers = []

        super(ActivationsTask, self).__init__(**kwargs)

    @override
    def name(self):
        return 'Infer Pretrained Model'

    @override
    def __getstate__(self):
        state = super(ActivationsTask, self).__getstate__()
        if 'inference_log' in state:
            # don't save file handle
            del state['inference_log']
        return state

    @override
    def __setstate__(self, state):
        super(ActivationsTask, self).__setstate__(state)

    @override
    def before_run(self):
        super(ActivationsTask, self).before_run()
        # create log file
        self.inference_log = open(self.path(self.inference_log_file), 'a')

    @override
    def process_output(self, line):
        self.inference_log.write('%s\n' % line)
        self.inference_log.flush()

        timestamp, level, message = self.preprocess_output_digits(line)
        if not message:
            return False

        # progress
        match = re.match(r'Processed (\d+)\/(\d+)', message)
        if match:
            self.progress = float(match.group(1))/int(match.group(2))
            return True

        # path to inference data
        match = re.match(r'Saved data to (.*)', message)
        if match:
            self.inference_data_filename = match.group(1).strip()
            return True

        return False

    @override
    def after_run(self):
        super(ActivationsTask, self).after_run()
        self.inference_log.close()

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

    @override
    def task_arguments(self, resources, env):

        if self.pretrained_model.framework == "caffe":
            inference_tool_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(digits.__file__))), 'tools', 'get_activations.py')
        else:
            inference_tool_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(digits.__file__))), 'tools', 'get_torch_activations.py')

        args = [sys.executable,
            inference_tool_path,
            self.image_path,
            self.pretrained_model.dir()
            ]

        args.append('--model_def_path=%s' % self.deploy_path)
        args.append('--weights_path=%s' % self.weights_path)
        args.append('--height=%s' % self.image_info["height"])
        args.append('--width=%s' % self.image_info["width"])
        args.append('--channels=%s' % self.image_info["image_type"])
        args.append('--resize_mode=%s' % self.image_info["resize_mode"])

        if self.gpu is not None:
            args.append('--gpu=%d' % self.gpu)

        return args
