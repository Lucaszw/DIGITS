# Copyright (c) 2014-2016, NVIDIA CORPORATION.  All rights reserved.
from __future__ import absolute_import

import json
import os

from ..job import ModelJob
from digits.utils import subclass, override

# NOTE: Increment this everytime the pickled object changes
PICKLE_VERSION = 1

class ImageModelJob(ModelJob):
    """
    A Job that creates an image model
    """

    def __init__(self, **kwargs):
        """
        """
        super(ImageModelJob, self).__init__(**kwargs)
        self.pickver_job_model_image = PICKLE_VERSION

    @override
    def get_job_stats_as_json_string(self,epoch=-1):
        task = self.train_task()

        stats = {
            "job id": self.id(),
            "creation time": self.status_history[0][1],
            "username": self.username,
        }

        stats.update(task.get_task_stats(epoch))

        return json.dumps(stats)
