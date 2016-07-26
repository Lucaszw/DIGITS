# Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.
from __future__ import absolute_import

import digits.frameworks
from digits.job import Job
from digits.utils import subclass, override
from digits.inference.tasks import ActivationsTask

@subclass
class ActivationsJob(Job):
    """
    A Job that exercises the forward pass of a neural network
    """

    def __init__(self, pretrained_model, image_path, **kwargs):
        """
        Arguments:
        pretrained_model -- job object associated with pretrained_model to perform inference on
        images -- list of image paths to perform inference on
        """
        super(ActivationsJob, self).__init__(persistent = False, **kwargs)
        self.pretrained_model = pretrained_model
        # create inference task
        self.tasks.append(ActivationsTask(
            pretrained_model,
            image_path,
            job_dir = self.dir()
            )
        )

    @override
    def __getstate__(self):
        fields_to_save = ['_id', '_name']
        full_state = super(ActivationsJob, self).__getstate__()
        state_to_save = {}
        for field in fields_to_save:
            state_to_save[field] = full_state[field]
        return state_to_save


    def inference_task(self):
        """Return the first and only InferenceTask for this job"""
        return [t for t in self.tasks if isinstance(t, tasks.ActivationsTask)][0]

    @override
    def __setstate__(self, state):
        super(ActivationsJob, self).__setstate__(state)

    def get_data(self):
        """Return inference data"""
        task = self.inference_task()
