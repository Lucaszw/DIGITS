import flask
import tempfile
import os
from digits import dataset, extensions, model, utils
from digits.webapp import app, scheduler
from digits.pretrained_model import PretrainedModelJob
from digits.utils.routing import request_wants_json, job_from_request
import werkzeug.exceptions

blueprint = flask.Blueprint(__name__, __name__)


@blueprint.route('/classify_one.json', methods=['POST'])
@blueprint.route('/classify_one', methods=['POST', 'GET'])
def classify_one():
    model_job = job_from_request()

    return str(isinstance(model_job, PretrainedModelJob))

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

    layers = 'none'
    if 'show_visualizations' in flask.request.form and flask.request.form['show_visualizations']:
        layers = 'all'

    # create inference job
    inference_job = ImageInferenceJob(
        username    = utils.auth.get_username(),
        name        = "Classify One Image",
        model       = model_job,
        images      = [image_path],
        epoch       = None,
        layers      = layers
        )


    return flask.request.files.tolist()

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

    # Create a temporary file to hold prototxt and caffemodel:
    prototxt = tempfile.mkstemp(suffix='.prototxt')
    flask.request.files['prototxt_file'].save(prototxt[1])
    prototxt_path = prototxt[1]
    os.close(prototxt[0])

    caffemodel = tempfile.mkstemp(suffix='.caffemodel')
    flask.request.files['caffemodel_file'].save(caffemodel[1])
    caffemodel_path = caffemodel[1]
    os.close(caffemodel[0])

    job = PretrainedModelJob(
        prototxt_path,
        caffemodel_path,
        username = utils.auth.get_username(),
        name = flask.request.form['job_name']
    )


    scheduler.add_job(job)

    return flask.redirect(flask.url_for('digits.views.home'))
