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
import subprocess

@subclass
class UploadPretrainedModelTask(Task):
    """
    A task for uploading pretrained models
    """

    def __init__(self, prototxt_path, caffemodel_path, **kwargs):
        """
        Arguments:
        prototxt_path -- path to deploy.prototxt
        caffemodel_path  -- path to caffemodel
        """
        # memorize parameters
        self.prototxt_path = prototxt_path
        self.caffemodel_path = caffemodel_path

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

    @override
    def run(self, resources):
        env = os.environ.copy()
        args1  = ["mv", self.prototxt_path, self.job_dir+"/deploy.prototxt"]
        args2  = ["mv", self.caffemodel_path, self.job_dir+"/model.caffemodel"]
        p = subprocess.Popen(args1)
        p = subprocess.Popen(args2)
        self.status = Status.DONE
