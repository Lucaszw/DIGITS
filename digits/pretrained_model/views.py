import flask
import tempfile
import os
import h5py
import numpy as np
from digits import dataset, extensions, model, utils
from digits.webapp import app, scheduler
from digits.pretrained_model import PretrainedModelJob
from digits.utils.routing import request_wants_json, job_from_request
from digits import utils
import werkzeug.exceptions

blueprint = flask.Blueprint(__name__, __name__)

def get_tempfile(file, suffix):
    temp = tempfile.mkstemp(suffix=suffix)
    file.save(temp[1])
    path = temp[1]
    os.close(temp[0])
    return path

@utils.auth.requires_login
@blueprint.route('/new', methods=['POST'])
def new():
    """
    Upload a pretrained model
    """
    weights_path   = get_tempfile(flask.request.files['weights_file'],".caffemodel")
    model_def_path = get_tempfile(flask.request.files['model_def_file'],".prototxt")
    labels_path = None

    if str(flask.request.files['labels_file'].filename) is not '':
        labels_path = get_tempfile(flask.request.files['labels_file'],".txt")

    job = PretrainedModelJob(
        weights_path,
        model_def_path,
        labels_path,
        flask.request.form['framework'],
        username = utils.auth.get_username(),
        name = flask.request.form['job_name'],
    )
    scheduler.add_job(job)

    return flask.redirect(flask.url_for('digits.views.home'))
