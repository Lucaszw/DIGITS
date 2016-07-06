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

    def __init__(self, caffemodel_path, deploy_path, train_val_path, solver_path, labels_path=None, **kwargs):
        """
        Arguments:
        prototxt_path -- path to deploy.prototxt
        caffemodel_path  -- path to caffemodel
        """
        # memorize parameters
        self.caffemodel_path = caffemodel_path
        self.deploy_path = deploy_path
        self.train_val_path = train_val_path
        self.solver_path = solver_path
        self.labels_path = labels_path
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

    def move_file(self,input, output):
        args  = ["mv", input, self.job_dir+"/"+output]
        p = subprocess.Popen(args)

    @override
    def run(self, resources):
        env = os.environ.copy()
        self.move_file(self.caffemodel_path, "model.caffemodel")

        if self.deploy_path is not None:
            self.move_file(self.deploy_path, "deploy.prototxt")

        if self.train_val_path is not None:
            self.move_file(self.train_val_path, "train_val.prototxt")

        if self.solver_path is not None:
            self.move_file(self.solver_path, "solver.prototxt")

        if self.labels_path is not None:
            self.move_file(self.labels_path, "labels.txt")

        self.status = Status.DONE
