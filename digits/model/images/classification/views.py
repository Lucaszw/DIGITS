# Copyright (c) 2014-2016, NVIDIA CORPORATION.  All rights reserved.
from __future__ import absolute_import

import os
from subprocess import call, Popen
import random
import re
import tempfile
import flask
import numpy as np
import werkzeug.exceptions

from .forms import ImageClassificationModelForm
from .job import ImageClassificationModelJob
import digits
from digits import frameworks
from digits import utils
from digits.config import config_value
from digits.dataset import ImageClassificationDatasetJob
from digits.inference import ImageInferenceJob
from digits.status import Status
from digits.utils import filesystem as fs
from digits.utils.forms import fill_form_if_cloned, save_form_to_job
from digits.utils.routing import request_wants_json, job_from_request
from digits.webapp import app, scheduler

blueprint = flask.Blueprint(__name__, __name__)

# TODO: Move this somewhere else!
import h5py
from datetime import datetime


"""
Read image list
"""
def read_image_list(image_list, image_folder, num_test_images):
    paths = []
    ground_truths = []

    for line in image_list.readlines():
        line = line.strip()
        if not line:
            continue

        # might contain a numerical label at the end
        match = re.match(r'(.*\S)\s+(\d+)$', line)
        if match:
            path = match.group(1)
            ground_truth = int(match.group(2))
        else:
            path = line
            ground_truth = None

        if not utils.is_url(path) and image_folder and not os.path.isabs(path):
            path = os.path.join(image_folder, path)
        paths.append(path)
        ground_truths.append(ground_truth)

        if num_test_images is not None and len(paths) >= num_test_images:
            break
    return paths, ground_truths

@blueprint.route('/new', methods=['GET'])
@utils.auth.requires_login
def new():
    """
    Return a form for a new ImageClassificationModelJob
    """
    form = ImageClassificationModelForm()
    form.dataset.choices = get_datasets()
    form.standard_networks.choices = get_standard_networks()
    form.standard_networks.default = get_default_standard_network()
    form.previous_networks.choices = get_previous_networks()

    prev_network_snapshots = get_previous_network_snapshots()

    ## Is there a request to clone a job with ?clone=<job_id>
    fill_form_if_cloned(form)

    return flask.render_template('models/images/classification/new.html',
            form = form,
            frameworks = frameworks.get_frameworks(),
            previous_network_snapshots = prev_network_snapshots,
            previous_networks_fullinfo = get_previous_networks_fulldetails(),
            multi_gpu = config_value('caffe_root')['multi_gpu'],
            )

@blueprint.route('.json', methods=['POST'])
@blueprint.route('', methods=['POST'], strict_slashes=False)
@utils.auth.requires_login(redirect=False)
def create():
    """
    Create a new ImageClassificationModelJob

    Returns JSON when requested: {job_id,name,status} or {errors:[]}
    """
    form = ImageClassificationModelForm()
    form.dataset.choices = get_datasets()
    form.standard_networks.choices = get_standard_networks()
    form.standard_networks.default = get_default_standard_network()
    form.previous_networks.choices = get_previous_networks()

    prev_network_snapshots = get_previous_network_snapshots()

    ## Is there a request to clone a job with ?clone=<job_id>
    fill_form_if_cloned(form)

    if not form.validate_on_submit():
        if request_wants_json():
            return flask.jsonify({'errors': form.errors}), 400
        else:
            return flask.render_template('models/images/classification/new.html',
                    form = form,
                    frameworks = frameworks.get_frameworks(),
                    previous_network_snapshots = prev_network_snapshots,
                    previous_networks_fullinfo = get_previous_networks_fulldetails(),
                    multi_gpu = config_value('caffe_root')['multi_gpu'],
                    ), 400

    datasetJob = scheduler.get_job(form.dataset.data)
    if not datasetJob:
        raise werkzeug.exceptions.BadRequest(
                'Unknown dataset job_id "%s"' % form.dataset.data)

    # sweeps will be a list of the the permutations of swept fields
    # Get swept learning_rate
    sweeps = [{'learning_rate': v} for v in form.learning_rate.data]
    add_learning_rate = len(form.learning_rate.data) > 1

    # Add swept batch_size
    sweeps = [dict(s.items() + [('batch_size', bs)]) for bs in form.batch_size.data for s in sweeps[:]]
    add_batch_size = len(form.batch_size.data) > 1
    n_jobs = len(sweeps)

    jobs = []
    for sweep in sweeps:
        # Populate the form with swept data to be used in saving and
        # launching jobs.
        form.learning_rate.data = sweep['learning_rate']
        form.batch_size.data = sweep['batch_size']

        # Augment Job Name
        extra = ''
        if add_learning_rate:
            extra += ' learning_rate:%s' % str(form.learning_rate.data[0])
        if add_batch_size:
            extra += ' batch_size:%d' % form.batch_size.data[0]

        job = None
        try:
            job = ImageClassificationModelJob(
                    username    = utils.auth.get_username(),
                    name        = form.model_name.data + extra,
                    dataset_id  = datasetJob.id(),
                    )
            # get handle to framework object
            fw = frameworks.get_framework_by_id(form.framework.data)

            pretrained_model = None
            if form.method.data == 'standard':
                found = False

                # can we find it in standard networks?
                network_desc = fw.get_standard_network_desc(form.standard_networks.data)
                if network_desc:
                    found = True
                    network = fw.get_network_from_desc(network_desc)

                if not found:
                    raise werkzeug.exceptions.BadRequest(
                            'Unknown standard model "%s"' % form.standard_networks.data)
            elif form.method.data == 'previous':
                old_job = scheduler.get_job(form.previous_networks.data)
                if not old_job:
                    raise werkzeug.exceptions.BadRequest(
                            'Job not found: %s' % form.previous_networks.data)

                use_same_dataset = (old_job.dataset_id == job.dataset_id)
                network = fw.get_network_from_previous(old_job.train_task().network, use_same_dataset)

                for choice in form.previous_networks.choices:
                    if choice[0] == form.previous_networks.data:
                        epoch = float(flask.request.form['%s-snapshot' % form.previous_networks.data])
                        if epoch == 0:
                            pass
                        elif epoch == -1:
                            pretrained_model = old_job.train_task().pretrained_model
                        else:
                            for filename, e in old_job.train_task().snapshots:
                                if e == epoch:
                                    pretrained_model = filename
                                    break

                            if pretrained_model is None:
                                raise werkzeug.exceptions.BadRequest(
                                        "For the job %s, selected pretrained_model for epoch %d is invalid!"
                                        % (form.previous_networks.data, epoch))
                            if not (os.path.exists(pretrained_model)):
                                raise werkzeug.exceptions.BadRequest(
                                        "Pretrained_model for the selected epoch doesn't exists. May be deleted by another user/process. Please restart the server to load the correct pretrained_model details")
                        break

            elif form.method.data == 'custom':
                network = fw.get_network_from_desc(form.custom_network.data)
                pretrained_model = form.custom_network_snapshot.data.strip()
            else:
                raise werkzeug.exceptions.BadRequest(
                        'Unrecognized method: "%s"' % form.method.data)

            policy = {'policy': form.lr_policy.data}
            if form.lr_policy.data == 'fixed':
                pass
            elif form.lr_policy.data == 'step':
                policy['stepsize'] = form.lr_step_size.data
                policy['gamma'] = form.lr_step_gamma.data
            elif form.lr_policy.data == 'multistep':
                policy['stepvalue'] = form.lr_multistep_values.data
                policy['gamma'] = form.lr_multistep_gamma.data
            elif form.lr_policy.data == 'exp':
                policy['gamma'] = form.lr_exp_gamma.data
            elif form.lr_policy.data == 'inv':
                policy['gamma'] = form.lr_inv_gamma.data
                policy['power'] = form.lr_inv_power.data
            elif form.lr_policy.data == 'poly':
                policy['power'] = form.lr_poly_power.data
            elif form.lr_policy.data == 'sigmoid':
                policy['stepsize'] = form.lr_sigmoid_step.data
                policy['gamma'] = form.lr_sigmoid_gamma.data
            else:
                raise werkzeug.exceptions.BadRequest(
                        'Invalid learning rate policy')

            if config_value('caffe_root')['multi_gpu']:
                if form.select_gpus.data:
                    selected_gpus = [str(gpu) for gpu in form.select_gpus.data]
                    gpu_count = None
                elif form.select_gpu_count.data:
                    gpu_count = form.select_gpu_count.data
                    selected_gpus = None
                else:
                    gpu_count = 1
                    selected_gpus = None
            else:
                if form.select_gpu.data == 'next':
                    gpu_count = 1
                    selected_gpus = None
                else:
                    selected_gpus = [str(form.select_gpu.data)]
                    gpu_count = None

            # Python Layer File may be on the server or copied from the client.
            fs.copy_python_layer_file(
                bool(form.python_layer_from_client.data),
                job.dir(),
                (flask.request.files[form.python_layer_client_file.name]
                 if form.python_layer_client_file.name in flask.request.files
                 else ''), form.python_layer_server_file.data)

            job.tasks.append(fw.create_train_task(
                        job = job,
                        dataset = datasetJob,
                        train_epochs = form.train_epochs.data,
                        snapshot_interval = form.snapshot_interval.data,
                        learning_rate = form.learning_rate.data[0],
                        lr_policy = policy,
                        gpu_count = gpu_count,
                        selected_gpus = selected_gpus,
                        batch_size = form.batch_size.data[0],
                        batch_accumulation = form.batch_accumulation.data,
                        val_interval = form.val_interval.data,
                        pretrained_model = pretrained_model,
                        crop_size = form.crop_size.data,
                        use_mean = form.use_mean.data,
                        network = network,
                        random_seed = form.random_seed.data,
                        solver_type = form.solver_type.data,
                        shuffle = form.shuffle.data,
                        )
                    )

            ## Save form data with the job so we can easily clone it later.
            save_form_to_job(job, form)

            jobs.append(job)
            scheduler.add_job(job)
            if n_jobs == 1:
                if request_wants_json():
                    return flask.jsonify(job.json_dict())
                else:
                    return flask.redirect(flask.url_for('digits.model.views.show', job_id=job.id()))

        except:
            if job:
                scheduler.delete_job(job)
            raise

    if request_wants_json():
        return flask.jsonify(jobs=[job.json_dict() for job in jobs])

    # If there are multiple jobs launched, go to the home page.
    return flask.redirect('/')

def show(job, related_jobs=None):
    """
    Called from digits.model.views.models_show()
    """
    return flask.render_template('models/images/classification/show.html', job=job, framework_ids = [fw.get_id() for fw in frameworks.get_frameworks()], related_jobs=related_jobs)

@blueprint.route('/large_graph', methods=['GET'])
def large_graph():
    """
    Show the loss/accuracy graph, but bigger
    """
    job = job_from_request()

    return flask.render_template('models/images/classification/large_graph.html', job=job)


@blueprint.route('/layer_visualizations', methods=['POST', 'GET'])
def layer_visualizations():
    # Get initial visualizations from job request:
    model_job = job_from_request()
    task = model_job.train_task()
    net       = task.get_net(None,0)

    o = "digits/layer_outputs/"
    # Copy files to layer_outputs (as this is the directory visualizations
    # will read from:
    prototxt_path = o+"deploy.prototxt"
    call(["cp", task.get_depoly_prototxt(), prototxt_path])
    call(["cp", task.get_caffemodel(), o+"model.caffemodel"])

    # Add input param force_backward: true to enable deconv:
    with file(prototxt_path, 'r') as original: data = original.read()
    if 'force_backward' not in data:
        with file(prototxt_path, 'w') as modified: modified.write("force_backward: true\n" + data)

    # Get list of pre-trained models available for visualizations:
    lst = os.listdir(o+"jobs")
    pretrained = []
    for path in lst:
        f = h5py.File(o+'jobs/'+path+'/vis_data.hdf5')
        jobname = f['jobname'].attrs['jobname']
        pretrained.append({'jobname': jobname, 'path': path})

    # Render view:
    prototxt  = open(prototxt_path,'r').read()
    return flask.render_template('models/images/classification/layer_visualizations.html',
            model_job  = model_job,
            prototxt   = prototxt,
            pretrained = pretrained
            )

@blueprint.route('/run_model.json', methods=['POST', 'GET'])
@blueprint.route('/run_model',  methods=['POST', 'GET'])
def run_model():
    """
    Save a pre-trained model
    prototxt <file> -- deploy.prototxt file
    caffemodel <file> -- ***.caffemodel
    jobname <form> -- what to save job as
    """

    timestamp = datetime.now().strftime('%Y-%m-%d%H-%M-%S')

    # Get prototxt (model definition) from form:
    prototxt = tempfile.mkstemp(suffix='.prototxt')
    flask.request.files['prototxt'].save(prototxt[1])
    prototxt_path = prototxt[1]
    os.close(prototxt[0])

    # Get weights from form:
    caffemodel = tempfile.mkstemp(suffix='.caffemodel')
    flask.request.files['weights'].save(caffemodel[1])
    caffemodel_path = caffemodel[1]
    os.close(caffemodel[0])

    outputs_path = os.path.abspath(digits.__path__[0])+"/layer_outputs/"

    # Move the model definition and weights into the layer_outputs folder:
    call(["mv", prototxt_path, outputs_path+"deploy.prototxt"])
    call(["mv", caffemodel_path, outputs_path+"model.caffemodel"])

    # Add force_backward to input parameters for deconv:
    filename = outputs_path+"deploy.prototxt"
    with file(filename, 'r') as original: data = original.read()
    if 'force_backward' not in data:
        with file(filename, 'w') as modified: modified.write("force_backward: true\n" + data)

    o_proto = outputs_path+"jobs/"+timestamp+"/deploy.prototxt"
    o_caffmodel = outputs_path+"jobs/"+timestamp+"/model.caffemodel"

    # Add prototxt and caffe model to jobs directory:
    call(["mkdir", outputs_path+"jobs/"+timestamp])
    call(["cp", outputs_path+"deploy.prototxt", o_proto])
    call(["cp", outputs_path+"model.caffemodel", o_caffmodel])

    # Write a database to hold vis information:
    f = h5py.File('digits/layer_outputs/jobs/'+timestamp+'/vis_data.hdf5','w-')
    dset = f.create_dataset("jobname", (1,) , dtype="i")
    dset.attrs['jobname'] = flask.request.form['jobname']
    dset.attrs['timestamp'] = timestamp

    # Save Weights:
    p = Popen(["python", outputs_path+"get_weights.py", o_proto, o_caffmodel, outputs_path+"jobs/"+timestamp])
    p.wait()

    # Render new model def in view:
    prototxt = open(filename,'r').read()

    return flask.jsonify({'data': {"prototxt": prototxt}})


@blueprint.route('/load_pretrained_model.json', methods=['POST'])
@blueprint.route('/load_pretrained_model', methods=['POST', 'GET'])
def load_pretrained_model():
    """
    Load a pre-trained model
    path <Args> -- path to job
    """
    path            = "jobs/"+flask.request.args['path']  + "/"
    outputs_path    = os.path.abspath(digits.__path__[0]) + "/layer_outputs/"
    job_path        = outputs_path + path
    prototxt_path   = job_path+"deploy.prototxt"
    caffemodel_path = job_path+"model.caffemodel"

    # Get all the images stored for this job:
    f = h5py.File(job_path+'/activations.hdf5','a')

    image_data = []
    for key in f:
        image_data.append({"key": key , "img": f[key]['data'][:].tolist()})

    # Get last backprop job:
    b = h5py.File(job_path+'/backprops.hdf5','a')

    if "0" in b:
        backprop_info = {
            "data": b["0"]["info"][:].tolist(),
            "attrs": dict(b["0"]["info"].attrs.items())
        }
    else:
        backprop_info = {}

    # Render new model def in view:
    prototxt = open(prototxt_path,'r').read()
    return flask.jsonify({'data': {"prototxt": prototxt, "backprop": backprop_info}, 'images': image_data})


@blueprint.route('/get_backprop_from_neuron_in_layer.json', methods=['POST'])
@blueprint.route('/get_backprop_from_neuron_in_layer', methods=['POST', 'GET'])
def get_backprop_from_neuron_in_layer():
    """
    Runs backprop from the a neuron in a specified layer
    path <Args>         -- path to job
    image_key <Args>    -- image_key in activations.h5py (0..N=num images)
    layer_name <Args>   -- layer that neuron resides in
    neuron_index <Args> -- index of selected neuron
    """

    # Get layer and neuron from inputs:
    delchars = ''.join(c for c in map(chr, range(256)) if not c.isalnum())
    o = os.path.abspath(digits.__path__[0])+"/layer_outputs/"

    # Get params from flask args:
    path          = o+"jobs/"+flask.request.args['path']
    layer_name    = str(flask.request.args['layer_name'])
    neuron_index  = str(flask.request.args['neuron_index'])
    image_key     = str(flask.request.args['image_key'])

    # Run the backprop script:
    p = Popen(["python", o+"get_backprops.py", path, layer_name, neuron_index])
    p.wait()

    # Return data for this layer:
    return flask.jsonify(backprops(path,image_key,layer_name))


@blueprint.route('/deconv_neuron_in_layer.json', methods=['POST'])
@blueprint.route('/deconv_neuron_in_layer', methods=['POST', 'GET'])
def deconv_neuron_in_layer():
    """
    Sends the neuron features from a given layer into Deconv Network
    and returns the the output of the 'data' layer
    path <Args> -- path to job
    image_key <Args> --  key to image ("0" .. num of images in db)
    layer_name <Args> -- layer that neuron resides in
    layer_type <Args> -- layer that neuron resides in
    neuron_index <Args> -- index of selected neuron
    """
    delchars = ''.join(c for c in map(chr, range(256)) if not c.isalnum())
    o = os.path.abspath(digits.__path__[0])+"/layer_outputs/"

    # Get params from flask args:
    path          = o+"jobs/"+flask.request.args['path']
    image_key     = flask.request.args['image_key']
    layer_name    = str(flask.request.args['layer_name'])
    neuron_index  = str(flask.request.args['neuron_index'])

    # Run the deconvolution script
    p = Popen(["python", o+"get_deconv.py", path, image_key, layer_name, neuron_index])
    p.wait()

    # Get outputs:
    deconv_path  = o+"deconv/" + layer_name.translate(None,delchars)+".npy"
    data = np.load(deconv_path)

    return flask.jsonify({'data': data.tolist()})


@blueprint.route('/get_outputs.json', methods=['POST', 'GET'])
@blueprint.route('/get_outputs',  methods=['POST', 'GET'])
def get_outputs():
    """
    Return the outputs of weights and activations for a given layer
    path <Args> -- path to job
    layer_name <Args> -- name of layer
    image_key <Args> --  key to image ("0" .. num of images in db)
    """
    path = os.path.abspath(digits.__path__[0])+"/layer_outputs/jobs/"+ str(flask.request.args['path'])+"/"

    image_key  = flask.request.args['image_key']
    layer_name = flask.request.args['layer_name']
    layers = []
    layers.append({'type':'weights', 'data': weights(path,layer_name)})
    layers.append({'type':'activations', 'data': activations(path,image_key,layer_name)})
    layers.append({'type':'backprops', 'data': backprops(path,image_key,layer_name)["data"]})

    return flask.jsonify({'layers': layers})

@blueprint.route('/get_weights.json', methods=['POST', 'GET'])
@blueprint.route('/get_weights',  methods=['POST', 'GET'])
def get_weights():
    """
    Returns the weights for a selected layer
    path <Args> -- path to job
    layer_name <Args> -- name of layer
    """
    # Get Path and Layer Name from input arguments:
    path = os.path.abspath(digits.__path__[0])+"/layer_outputs/jobs/"+ str(flask.request.args['path'])+"/"
    layer_name  = str(flask.request.args['layer_name'])

    return flask.jsonify({'data':  weights(path, layer_name)})

@blueprint.route('/get_activations.json', methods=['POST', 'GET'])
@blueprint.route('/get_activations',  methods=['POST', 'GET'])
def get_activations():
    """
    Return the activations for a specific layer and image
    path <Args> -- path to job
    layer_name <Args> -- name of layer
    image_key <Args> --  key to image ("0" .. num of images in db)
    """
    # Get Path and Layer Name from input arguments:
    path = os.path.abspath(digits.__path__[0])+"/layer_outputs/jobs/"+ str(flask.request.args['path'])+"/"
    layer_name  = str(flask.request.args['layer_name'])
    image_key   = str(flask.request.args['image_key'])

    return flask.jsonify({'data': activations(path,image_key,layer_name)})

@blueprint.route('/get_backprops.json', methods=['POST', 'GET'])
@blueprint.route('/get_backprops',  methods=['POST', 'GET'])
def get_backprops():
    """
    Return the activations for a specific layer and image
    path <Args> -- path to job
    layer_name <Args> -- name of layer
    image_key <Args> --  key to image ("0" .. num of images in db)
    """
    # Get Path and Layer Name from input arguments:
    path = os.path.abspath(digits.__path__[0])+"/layer_outputs/jobs/"+ str(flask.request.args['path'])+"/"
    layer_name  = str(flask.request.args['layer_name'])
    image_key   = str(flask.request.args['image_key'])

    return flask.jsonify({'data': backprops(path,image_key,layer_name)["data"]})


def backprops(path,image_key,layer_name):
    # Read backprops file, and group containing activations for given image:
    f = h5py.File(path+'/backprops.hdf5','a')
    data = []
    info = {}
    if image_key in f:
        if layer_name in f[image_key]:

            s = int(f[image_key][layer_name].shape[1]/40 +1)
            data = f[image_key][layer_name][:100,::s,::s].tolist()
            info = {
                "data":  f[image_key]["info"][:].tolist(),
                "attrs": dict(f[image_key]["info"].attrs.items())
            }

    return {"data": data, "info":info}

def activations(path,image_key,layer_name):
    # Read activations file, and group containing activations for given image:
    f = h5py.File(path+'activations.hdf5','a')
    data = []

    # Return activations of first 100 neurons:
    # TODO: Resolution should be an input parameter
    if image_key in f:
        grp = f[image_key]
        if layer_name in grp:
            s = int(f[image_key][layer_name].shape[1]/40 +1)
            data = grp[layer_name][:100,::s,::s].tolist()

    return data

def weights(path,layer_name):
    # Read weights file:
    f = h5py.File(path+'weights.hdf5','r')
    if layer_name in f:
        data = f[layer_name][:100].tolist()
    else:
        data = []

    return data

@blueprint.route('/send_params.json', methods=['POST', 'GET'])
@blueprint.route('/send_params',  methods=['POST', 'GET'])
def send_params():
    """
    Load image & Save activations
    load_default <Args> -- true/false (should load from current task or folder)
    job_path     <Args> -- path of job to save outputs to
    """
    # Set job path from flask input arguments
    o = os.path.abspath(digits.__path__[0])+"/layer_outputs/"
    path = o+"jobs/"+ str(flask.request.args['path'])

    # Store image file temporarily:
    outfile = tempfile.mkstemp(suffix='.png')
    flask.request.files['image_file'].save(outfile[1])
    image_path = outfile[1]
    os.close(outfile[0])

    # Save Activations:
    p = Popen(["python", o+"get_activations.py", image_path, path])
    p.wait()

    return flask.jsonify({'data': "success!"})

@blueprint.route('/classify_one.json', methods=['POST'])
@blueprint.route('/classify_one', methods=['POST', 'GET'])
def classify_one():
    """
    Classify one image and return the top 5 classifications

    Returns JSON when requested: {predictions: {category: confidence,...}}
    """
    model_job = job_from_request()

    remove_image_path = False
    image      = None
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

    epoch = None
    if 'snapshot_epoch' in flask.request.form:
        epoch = float(flask.request.form['snapshot_epoch'])

    layers = 'none'
    if 'show_visualizations' in flask.request.form and flask.request.form['show_visualizations']:
        layers = 'all'

    # create inference job
    inference_job = ImageInferenceJob(
                username    = utils.auth.get_username(),
                name        = "Classify One Image",
                model       = model_job,
                images      = [image_path],
                epoch       = epoch,
                layers      = layers
                )

    # schedule tasks
    scheduler.add_job(inference_job)

    # wait for job to complete
    inference_job.wait_completion()

    # retrieve inference data
    inputs, outputs, visualizations = inference_job.get_data()

    # set return status code
    status_code = 500 if inference_job.status == 'E' else 200

    # delete job
    scheduler.delete_job(inference_job)

    if remove_image_path:
        os.remove(image_path)

    predictions = []
    prototxt    = model_job.train_task().get_network_desc()


    if inputs is not None and len(inputs['data']) == 1:
        image = utils.image.embed_image_html(inputs['data'][0])

        # convert to class probabilities for viewing
        last_output_name, last_output_data = outputs.items()[-1]

        if len(last_output_data) == 1:
            scores = last_output_data[0].flatten()
            indices = (-scores).argsort()
            labels = model_job.train_task().get_labels()
            predictions = []
            for i in indices:
                # ignore prediction if we don't have a label for the corresponding class
                # the user might have set the final fully-connected layer's num_output to
                # too high a value
                if i < len(labels):
                    predictions.append( (labels[i], scores[i]) )
            predictions = [(p[0], round(100.0*p[1],2)) for p in predictions[:5]]


    if request_wants_json():
        return flask.jsonify({'predictions': predictions}), status_code
    else:
        return flask.render_template('models/images/classification/classify_one.html',
                model_job       = model_job,
                job             = inference_job,
                image_src       = image,
                image_data      = inputs['data'][0],
                predictions     = predictions,
                visualizations  = visualizations,
                prototxt        = prototxt,
                total_parameters= sum(v['param_count'] for v in visualizations if v['vis_type'] == 'Weights'),
                ), status_code

@blueprint.route('/classify_many.json', methods=['POST'])
@blueprint.route('/classify_many', methods=['POST', 'GET'])
def classify_many():
    """
    Classify many images and return the top 5 classifications for each

    Returns JSON when requested: {classifications: {filename: [[category,confidence],...],...}}
    """
    model_job = job_from_request()

    image_list = flask.request.files.get('image_list')
    if not image_list:
        raise werkzeug.exceptions.BadRequest('image_list is a required field')

    if 'image_folder' in flask.request.form and flask.request.form['image_folder'].strip():
        image_folder = flask.request.form['image_folder']
        if not os.path.exists(image_folder):
            raise werkzeug.exceptions.BadRequest('image_folder "%s" does not exit' % image_folder)
    else:
        image_folder = None

    if 'num_test_images' in flask.request.form and flask.request.form['num_test_images'].strip():
        num_test_images = int(flask.request.form['num_test_images'])
    else:
        num_test_images = None

    epoch = None
    if 'snapshot_epoch' in flask.request.form:
        epoch = float(flask.request.form['snapshot_epoch'])

    paths, ground_truths = read_image_list(image_list, image_folder, num_test_images)

    # create inference job
    inference_job = ImageInferenceJob(
                username    = utils.auth.get_username(),
                name        = "Classify Many Images",
                model       = model_job,
                images      = paths,
                epoch       = epoch,
                layers      = 'none'
                )

    # schedule tasks
    scheduler.add_job(inference_job)

    # wait for job to complete
    inference_job.wait_completion()

    # retrieve inference data
    inputs, outputs, _ = inference_job.get_data()

    # set return status code
    status_code = 500 if inference_job.status == 'E' else 200

    # delete job
    scheduler.delete_job(inference_job)

    if outputs is not None and len(outputs) < 1:
        # an error occurred
        outputs = None

    if inputs is not None:
        # retrieve path and ground truth of images that were successfully processed
        paths = [paths[idx] for idx in inputs['ids']]
        ground_truths = [ground_truths[idx] for idx in inputs['ids']]

    # defaults
    classifications = None
    show_ground_truth = None
    top1_accuracy = None
    top5_accuracy = None
    confusion_matrix = None
    per_class_accuracy = None
    labels = None

    if outputs is not None:
        # convert to class probabilities for viewing
        last_output_name, last_output_data = outputs.items()[-1]
        if len(last_output_data) < 1:
            raise werkzeug.exceptions.BadRequest(
                    'Unable to classify any image from the file')

        scores = last_output_data
        # take top 5
        indices = (-scores).argsort()[:, :5]

        labels = model_job.train_task().get_labels()
        n_labels = len(labels)

        # remove invalid ground truth
        ground_truths = [x if x is not None and (0 <= x < n_labels) else None for x in ground_truths]

        # how many pieces of ground truth to we have?
        n_ground_truth = len([1 for x in ground_truths if x is not None])
        show_ground_truth = n_ground_truth > 0

        # compute classifications and statistics
        classifications = []
        n_top1_accurate = 0
        n_top5_accurate = 0
        confusion_matrix = np.zeros((n_labels,n_labels), dtype=np.dtype(int))
        for image_index, index_list in enumerate(indices):
            result = []
            if ground_truths[image_index] is not None:
                if ground_truths[image_index] == index_list[0]:
                    n_top1_accurate += 1
                if ground_truths[image_index] in index_list:
                    n_top5_accurate += 1
                if (0 <= ground_truths[image_index] < n_labels) and (0 <= index_list[0] < n_labels):
                   confusion_matrix[ground_truths[image_index], index_list[0]] += 1
            for i in index_list:
                # `i` is a category in labels and also an index into scores
                # ignore prediction if we don't have a label for the corresponding class
                # the user might have set the final fully-connected layer's num_output to
                # too high a value
                if i < len(labels):
                    result.append((labels[i], round(100.0*scores[image_index, i],2)))
            classifications.append(result)

        # accuracy
        if show_ground_truth:
            top1_accuracy = round(100.0 * n_top1_accurate / n_ground_truth, 2)
            top5_accuracy = round(100.0 * n_top5_accurate / n_ground_truth, 2)
            per_class_accuracy = []
            for x in xrange(n_labels):
                n_examples = sum(confusion_matrix[x])
                per_class_accuracy.append(round(100.0 * confusion_matrix[x,x] / n_examples, 2) if n_examples > 0 else None)
        else:
            top1_accuracy = None
            top5_accuracy = None
            per_class_accuracy = None

        # replace ground truth indices with labels
        ground_truths = [labels[x] if x is not None and (0 <= x < n_labels ) else None for x in ground_truths]

    if request_wants_json():
        joined = dict(zip(paths, classifications))
        return flask.jsonify({'classifications': joined}), status_code
    else:
        return flask.render_template('models/images/classification/classify_many.html',
                model_job          = model_job,
                job                = inference_job,
                paths              = paths,
                classifications    = classifications,
                show_ground_truth  = show_ground_truth,
                ground_truths      = ground_truths,
                top1_accuracy      = top1_accuracy,
                top5_accuracy      = top5_accuracy,
                confusion_matrix   = confusion_matrix,
                per_class_accuracy = per_class_accuracy,
                labels             = labels,
                ), status_code

@blueprint.route('/top_n', methods=['POST'])
def top_n():
    """
    Classify many images and show the top N images per category by confidence
    """
    model_job = job_from_request()

    image_list = flask.request.files['image_list']
    if not image_list:
        raise werkzeug.exceptions.BadRequest('File upload not found')

    epoch = None
    if 'snapshot_epoch' in flask.request.form:
        epoch = float(flask.request.form['snapshot_epoch'])
    if 'top_n' in flask.request.form and flask.request.form['top_n'].strip():
        top_n = int(flask.request.form['top_n'])
    else:
        top_n = 9

    if 'image_folder' in flask.request.form and flask.request.form['image_folder'].strip():
        image_folder = flask.request.form['image_folder']
        if not os.path.exists(image_folder):
            raise werkzeug.exceptions.BadRequest('image_folder "%s" does not exit' % image_folder)
    else:
        image_folder = None

    if 'num_test_images' in flask.request.form and flask.request.form['num_test_images'].strip():
        num_test_images = int(flask.request.form['num_test_images'])
    else:
        num_test_images = None

    paths, _ = read_image_list(image_list, image_folder, num_test_images)

    # create inference job
    inference_job = ImageInferenceJob(
                username    = utils.auth.get_username(),
                name        = "TopN Image Classification",
                model       = model_job,
                images      = paths,
                epoch       = epoch,
                layers      = 'none'
                )

    # schedule tasks
    scheduler.add_job(inference_job)

    # wait for job to complete
    inference_job.wait_completion()

    # retrieve inference data
    inputs, outputs, _ = inference_job.get_data()

    # delete job
    scheduler.delete_job(inference_job)

    results = None
    if outputs is not None and len(outputs) > 0:
        # convert to class probabilities for viewing
        last_output_name, last_output_data = outputs.items()[-1]
        scores = last_output_data

        if scores is None:
            raise RuntimeError('An error occured while processing the images')

        labels = model_job.train_task().get_labels()
        images = inputs['data']
        indices = (-scores).argsort(axis=0)[:top_n]
        results = []
        # Can't have more images per category than the number of images
        images_per_category = min(top_n, len(images))
        # Can't have more categories than the number of labels or the number of outputs
        n_categories = min(indices.shape[1], len(labels))
        for i in xrange(n_categories):
            result_images = []
            for j in xrange(images_per_category):
                result_images.append(images[indices[j][i]])
            results.append((
                    labels[i],
                    utils.image.embed_image_html(
                        utils.image.vis_square(np.array(result_images),
                            colormap='white')
                        )
                    ))

    return flask.render_template('models/images/classification/top_n.html',
            model_job       = model_job,
            job             = inference_job,
            results         = results,
            )

def get_datasets():
    return [(j.id(), j.name()) for j in sorted(
        [j for j in scheduler.jobs.values() if isinstance(j, ImageClassificationDatasetJob) and (j.status.is_running() or j.status == Status.DONE)],
        cmp=lambda x,y: cmp(y.id(), x.id())
        )
        ]

def get_standard_networks():
    return [
            ('lenet', 'LeNet'),
            ('alexnet', 'AlexNet'),
            #('vgg-16', 'VGG (16-layer)'), #XXX model won't learn
            ('googlenet', 'GoogLeNet'),
            ]

def get_default_standard_network():
    return 'alexnet'

def get_previous_networks():
    return [(j.id(), j.name()) for j in sorted(
        [j for j in scheduler.jobs.values() if isinstance(j, ImageClassificationModelJob)],
        cmp=lambda x,y: cmp(y.id(), x.id())
        )
        ]

def get_previous_networks_fulldetails():
    return [(j) for j in sorted(
        [j for j in scheduler.jobs.values() if isinstance(j, ImageClassificationModelJob)],
        cmp=lambda x,y: cmp(y.id(), x.id())
        )
        ]

def get_previous_network_snapshots():
    prev_network_snapshots = []
    for job_id, _ in get_previous_networks():
        job = scheduler.get_job(job_id)
        e = [(0, 'None')] + [(epoch, 'Epoch #%s' % epoch)
                for _, epoch in reversed(job.train_task().snapshots)]
        if job.train_task().pretrained_model:
            e.insert(0, (-1, 'Previous pretrained model'))
        prev_network_snapshots.append(e)
    return prev_network_snapshots
