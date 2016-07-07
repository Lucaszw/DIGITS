import flask
import tempfile
import os
import h5py
import numpy as np
from digits import dataset, extensions, model, utils
from digits.webapp import app, scheduler
from digits.pretrained_model import PretrainedModelJob
from digits.visualizations.jobs import PretrainedModelInferenceJob
from digits.utils.routing import request_wants_json, job_from_request
from digits import utils
import werkzeug.exceptions

blueprint = flask.Blueprint(__name__, __name__)


def get_data(group):
    data = []
    for key in group.keys():
        if type(group[key]).__name__ is "Group":
            data += get_data(group[key])
        else:
            data.append({
                "data": utils.image.embed_image_html(group[key][:]),
                "attrs": dict(group[key].attrs.items())
                })

    return data

def activations_as_html(job_dir):

    f = h5py.File(job_dir+'/activations.hdf5','r')
    images = []

    for i in f.keys():
        grp = f[i]
        predictions = grp.attrs["predictions"]
        activations = get_data(grp)
        activations.sort(key= lambda item: item["attrs"]["i"])
        images.append({
        "predictions": predictions,
        "data": activations
        })
        # images.append(sorted(activations, key=lambda k: int(k["attrs"]['i'])))
    return images

def weights_as_html(job_dir):

    f = h5py.File(job_dir+'/weights.hdf5','r')
    weights = get_data(f)
    weights.sort(key=lambda item: item["attrs"]["i"])
    return weights

def get_tempfile(file, suffix):
    temp = tempfile.mkstemp(suffix=suffix)
    file.save(temp[1])
    path = temp[1]
    os.close(temp[0])
    return path

@blueprint.route('/classify_one', methods=['POST', 'GET'])
def classify_one():

    model_job = job_from_request()

    if flask.request.method == "POST":

        if 'image_path' in flask.request.form and flask.request.form['image_path']:
            image_path = flask.request.form['image_path']
        elif 'image_file' in flask.request.files and flask.request.files['image_file']:
            outfile = tempfile.mkstemp(suffix='.png')
            flask.request.files['image_file'].save(outfile[1])
            image_path = outfile[1]
            os.close(outfile[0])
            remove_image_path = True
        else:
            raise werkzeug.exceptions.BadRequest('must provide image_path or image_file')

        inference_job = PretrainedModelInferenceJob(
            model_job,
            [image_path],
            name     = "Classify One Image",
            username = utils.auth.get_username()
        )
        scheduler.add_job(inference_job)
        inference_job.wait_completion()
        scheduler.delete_job(inference_job)

    activations = activations_as_html(model_job.dir())[0]
    predictions = activations["predictions"]

    weights     = weights_as_html(model_job.dir())
    activations = activations["data"]


    return flask.render_template('visualizations/inference/classify_one.html',
            model_job   = model_job,
            weights     = weights,
            activations = activations,
            predictions = predictions
            )

@blueprint.route('/<job_id>.json', methods=['GET'])
@blueprint.route('/<job_id>', methods=['GET'])
def show(job_id):
    """
    Show a PretrainedModelJob

    Returns JSON when requested:
        {id, name, directory, status,...}
    """
    job = scheduler.get_job(job_id)
    if job is None:
        raise werkzeug.exceptions.NotFound('Job not found')

    if request_wants_json():
        return flask.jsonify(job.json_dict(True))
    else:
        return flask.render_template('pretrained_models/show.html', job=job)


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
