import flask
import tempfile
import tarfile
import zipfile
import json

import os
import shutil
import h5py
import numpy as np

from digits import dataset, extensions, model, utils
from digits.webapp import app, scheduler
from digits.pretrained_model import PretrainedModelJob
from digits.inference import ActivationsJob
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

def get_data_blob(id, activations):
    """ Get Data Blobs from Activation Blobs """
    return {"id": id, "data": activations["data"][:][0].tolist()}

def format_job_name(job):
    return {"name": job.name(), "id": job.id()}

@blueprint.route('/get_inference.json', methods=['GET'])
def get_inference():
    """ Return the weights and activations for a given layer """
    # TODO - Also get weights
    job = job_from_request()
    args = flask.request.args
    layer_name = args["layer_name"]
    image_id   = args["image_id"]
    range_min  = int(args["range_min"])
    range_max  = int(args["range_max"])

    data   = []
    num_units = 0
    if os.path.isfile(job.get_activations_path()):
        f = h5py.File(job.get_activations_path())
        if image_id in f:
            num_units = len(f[image_id][layer_name])
            data = f[image_id][layer_name][:][range_min:range_max].tolist()

    return flask.jsonify({"data": data, "length": num_units})

@blueprint.route('/upload_image.json', methods=['POST'])
def upload_image():
    model_job = job_from_request()
    image_path = get_tempfile(flask.request.files['image'],".png")

    activations_job = ActivationsJob(
        model_job,
        image_path,
        name     = "Upload Image",
        username = utils.auth.get_username()
    )

    scheduler.add_job(activations_job)
    activations_job.wait_completion()
    scheduler.delete_job(activations_job)

    f = h5py.File(model_job.get_activations_path())
    image_id = str(len(f.keys())-1)
    input_data = f[image_id]['data'][:][0]

    return flask.jsonify({"data": input_data.tolist(), "id": image_id})

@blueprint.route('/get_outputs.json', methods=['GET'])
def get_outputs():
    job  = scheduler.get_job(flask.request.args["job_id"])
    data = []

    if os.path.isfile(job.get_activations_path()):
        f = h5py.File(job.get_activations_path())
        data = [get_data_blob(k,v) for k,v in f.items()]

    return flask.jsonify({"model_def": job.get_model_def(), "images": data, "framework": job.framework})

@utils.auth.requires_login
@blueprint.route('/layer_visualizations', methods=['GET'])
def layer_visualizations():
    jobs = [format_job_name(x) for x in get_job_list(PretrainedModelJob,False)]
    return flask.render_template("pretrained_models/layer_visualizations.html",
            jobs=jobs)

@utils.auth.requires_login
@blueprint.route('/upload_archive', methods=['POST'])
def upload_archive():
    """
    Upload archive
    """
    files = flask.request.files
    archive_file = get_tempfile(files["archive"],".archive");

    if tarfile.is_tarfile(archive_file):
        archive = tarfile.open(archive_file,'r')
        names = archive.getnames()
    elif zipfile.is_zipfile(archive_file):
        archive = zipfile.ZipFile(archive_file, 'r')
        names = archive.namelist()
    else:
        return flask.jsonify({"status": "error"}), 500

    if "info.json" in names:

        # Create a temp directory to storce archive
        tempdir = tempfile.mkdtemp()
        archive.extractall(path=tempdir)

        with open(os.path.join(tempdir, "info.json")) as data_file:
            info = json.load(data_file)

        # Get path to files needed to be uploaded in directory
        weights_file = os.path.join(tempdir, info["snapshot file"])
        model_file   = os.path.join(tempdir, info["model file"])
        labels_file  = os.path.join(tempdir, info["labels file"])

        # Upload the Model:
        job = PretrainedModelJob(
            weights_file,
            model_file ,
            labels_file,
            info["framework"],
            info["image dimensions"][2],
            info["image resize mode"],
            info["image dimensions"][0],
            info["image dimensions"][1],
            username = utils.auth.get_username(),
            name = info["name"]
        )

        scheduler.add_job(job)
        job.wait_completion()

        # Delete temp directory
        shutil.rmtree(tempdir, ignore_errors=True)

        # Get Weights:
        weights_job = WeightsJob(
            job,
            name     = info['name'],
            username = utils.auth.get_username()
        )

        scheduler.add_job(weights_job)

        return flask.jsonify({"status": "success"}), 200
    else:
        return flask.jsonify({"status": "error"}), 500


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
        form["image_type"],
        form["resize_mode"],
        form["width"],
        form["height"],
        username = utils.auth.get_username(),
        name = flask.request.form['job_name']
    )

    scheduler.add_job(job)

    return flask.redirect(flask.url_for('digits.views.home', tab=3)), 302
