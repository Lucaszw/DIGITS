# Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.
from __future__ import absolute_import

from . import tasks
import digits.frameworks
from digits.job import Job
from digits.utils import subclass, override
from digits.pretrained_model.tasks import UploadPretrainedModelTask

@subclass
class PretrainedModelJob(Job):
    """
    A Job that uploads a pretrained model
    """

    def __init__(self, prototxt_path,caffemodel_path,**kwargs):
        super(PretrainedModelJob, self).__init__(persistent = False, **kwargs)
        self.tasks = []
        self.tasks.append(UploadPretrainedModelTask(
            prototxt_path,
            caffemodel_path,
            job_dir=self.dir()
        ))

    def get_deploy_prototxt(self):
        return self.dir()+"/deploy.prototxt"

    def get_caffemodel(self):
        return self.dir()+"/model.caffemodel"

    @override
    def __getstate__(self):
        fields_to_save = ['_id', '_name', 'username', 'tasks', 'status_history']
        full_state = super(PretrainedModelJob, self).__getstate__()
        state_to_save = {}
        for field in fields_to_save:
            state_to_save[field] = full_state[field]
        return state_to_save


    @override
    def __setstate__(self, state):
        super(PretrainedModelJob, self).__setstate__(state)
