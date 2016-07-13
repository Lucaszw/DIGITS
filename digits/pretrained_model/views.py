import flask
import tempfile
import os
import h5py
import numpy as np
from digits import dataset, extensions, model, utils
from digits.webapp import app, scheduler
from digits.pretrained_model import PretrainedModelJob
from digits.utils.routing import request_wants_json, job_from_request
from digits.views import get_job_list
from digits import utils
import werkzeug.exceptions

blueprint = flask.Blueprint(__name__, __name__)

def get_tempfile(f, suffix):
    temp = tempfile.mkstemp(suffix=suffix)
    f.save(temp[1])
    path = temp[1]
    os.close(temp[0])
    return path

def validate_caffe_files(files):
    """
    Upload a caffemodel
    """
    # Validate model weights:
    if str(files['weights_file'].filename) is '':
        raise werkzeug.exceptions.BadRequest('Missing weights file')
    elif files['weights_file'].filename.rsplit('.',1)[1] != "caffemodel" :
        raise werkzeug.exceptions.BadRequest('Weights must be a .caffemodel file')

    # Validate model definition:
    if str(files['model_def_file'].filename) is '':
        raise werkzeug.exceptions.BadRequest('Missing model definition file')
    elif files['model_def_file'].filename.rsplit('.',1)[1] != "prototxt" :
        raise werkzeug.exceptions.BadRequest('Model definition must be .prototxt file')

    weights_path   = get_tempfile(flask.request.files['weights_file'],".caffemodel")
    model_def_path = get_tempfile(flask.request.files['model_def_file'],".prototxt")

    return (weights_path, model_def_path)

def validate_torch_files(files):
    """
    Upload a torch model
    """
    # Validate model weights:
    if str(files['weights_file'].filename) is '':
        raise werkzeug.exceptions.BadRequest('Missing weights file')
    elif files['weights_file'].filename.rsplit('.',1)[1] != "t7" :
        raise werkzeug.exceptions.BadRequest('Weights must be a .t7 file')

    # Validate model definition:
    if str(files['model_def_file'].filename) is '':
        raise werkzeug.exceptions.BadRequest('Missing model definition file')
    elif files['model_def_file'].filename.rsplit('.',1)[1] != "lua" :
        raise werkzeug.exceptions.BadRequest('Model definition must be .lua file')

    weights_path   = get_tempfile(flask.request.files['weights_file'],".t7")
    model_def_path = get_tempfile(flask.request.files['model_def_file'],".lua")

    return (weights_path, model_def_path)

def format_job_name(job):
    return {"name": job.name(), "id": job.id()}

@blueprint.route('/get_outputs.json', methods=['GET'])
def get_outputs():
    job = scheduler.get_job(flask.request.args["job_id"])

    return flask.jsonify({"model_def": job.get_model_def()})

@utils.auth.requires_login
@blueprint.route('/layer_visualizations', methods=['GET'])
def layer_visualizations():
    jobs = [format_job_name(x) for x in get_job_list(PretrainedModelJob,False)]
    return flask.render_template("pretrained_models/layer_visualizations.html",
            jobs=jobs)

@utils.auth.requires_login
@blueprint.route('/new', methods=['POST'])
def new():
    """
    Upload a pretrained model
    """
    labels_path = None
    framework   = None

    form  = flask.request.form
    files = flask.request.files

    if 'framework' not in form:
        framework = "caffe"
    else:
        framework = form['framework']

    if 'job_name' not in flask.request.form:
        raise werkzeug.exceptions.BadRequest('Missing job name')
    elif str(flask.request.form['job_name']) is '':
        raise werkzeug.exceptions.BadRequest('Missing job name')

    if framework == "caffe":
        weights_path, model_def_path = validate_caffe_files(files)
    else:
        weights_path, model_def_path = validate_torch_files(files)

    if str(flask.request.files['labels_file'].filename) is not '':
        labels_path = get_tempfile(flask.request.files['labels_file'],".txt")

    job = PretrainedModelJob(
        weights_path,
        model_def_path,
        labels_path,
        framework,
        username = utils.auth.get_username(),
        name = flask.request.form['job_name']
    )

    scheduler.add_job(job)

    return flask.redirect(flask.url_for('digits.views.home')), 302
