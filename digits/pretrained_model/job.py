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

    def __init__(self, caffemodel_path, deploy_path, train_val_path, solver_path, labels_path=None,**kwargs):
        super(PretrainedModelJob, self).__init__(persistent = False, **kwargs)

        self.has_labels = labels_path is not None
        self.has_deploy = deploy_path is not None
        self.has_train_val = train_val_path is not None
        self.has_solver = solver_path is not None

        self.tasks = []
        self.tasks.append(UploadPretrainedModelTask(
            caffemodel_path,
            deploy_path,
            train_val_path,
            solver_path,
            labels_path,
            job_dir=self.dir()
        ))


    def get_caffemodel(self):
        return self.dir()+"/model.caffemodel"

    def get_deploy_prototxt(self):
        return self.dir()+"/deploy.prototxt"

    def get_train_val_prototxt(self):
        return self.dir()+"/train_val.prototxt"

    def get_solver_prototxt(self):
        return self.dir()+"/solver.prototxt"

    @override
    def job_type(self):
        return "Pretrained Model"

    @override
    def __getstate__(self):
        fields_to_save = ['_id', '_name', 'username', 'tasks', 'status_history', 'has_labels', 'has_deploy', 'has_train_val']
        full_state = super(PretrainedModelJob, self).__getstate__()
        state_to_save = {}
        for field in fields_to_save:
            state_to_save[field] = full_state[field]
        return state_to_save


    @override
    def __setstate__(self, state):
        super(PretrainedModelJob, self).__setstate__(state)
