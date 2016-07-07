# Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.
from __future__ import absolute_import
import os
import shutil

import digits
from digits.task import Task
from digits.utils import subclass, override

@subclass
class UploadPretrainedModelTask(Task):
    """
    A task for uploading pretrained models
    """

    def __init__(self, weights_path, model_def_path, labels_path=None, framework="caffe", **kwargs):
        """
        Arguments:
        weights_path -- path to model weights (**.caffemodel or ***.t7)
        model_def_path  -- path to model definition (**.prototxt or ***.lua)
        image_info -- a dictionary containing image_type, resize_mode, width, and height
        labels_path -- path to text file containing list of labels
        framework  -- framework of this job (ie caffe or torch)
        """
        self.weights_path = kwargs.pop('weights_path', None)
        self.model_def_path = kwargs.pop('model_def_path', None)
        self.image_info = kwargs.pop('image_info', None)
        self.labels_path = kwargs.pop('labels_path', None)
        self.mean_path = kwargs.pop('mean_path', None)
        self.framework = kwargs.pop('framework', None)

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
                return reserved_resources
        return None

    def move_file(self,input_file, output):
        shutil.copy(input_file, os.path.join(self.job_dir,output))

    def get_labels(self):
        labels = []
<<<<<<< HEAD
        if os.path.isfile(self.get_labels_path()):
            with open(self.get_labels_path()) as f:
=======
        if self.labels_path is not None:
            with open(self.job_dir+"/labels.txt") as f:
>>>>>>> Layer Visualizations And Weights for Caffe-Torch
                labels = f.readlines()
        return labels

    def get_model_def_path(self,as_json=False):
        """
        Get path to model definition
        """
        raise NotImplementedError('Please implement me')

    def get_weights_path(self):
        """
        Get path to model weights
        """
        raise NotImplementedError('Please implement me')

    def get_deploy_path(self):
        """
        Get path to file containing model def for deploy/visualization
        """
        raise NotImplementedError('Please implement me')

<<<<<<< HEAD
    def get_labels_path(self):
        return os.path.join(self.job_dir,"labels.txt")

    def get_mean_path(self):
        return os.path.join(self.job_dir,"mean.binaryproto")

=======
>>>>>>> Layer Visualizations And Weights for Caffe-Torch
    def write_deploy(self):
        """
        Write model definition for deploy/visualization
        """
        raise NotImplementedError('Please implement me')
