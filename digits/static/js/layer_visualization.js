var selected_image, selected_neuron, selected_layer, selected_type, job_path;
var load_default = true;
// Default to faded red,yellow,green,blue as color map (only applies if greyscale data):
var colormap = chroma.scale(['#541E8A','#3F84FE','#87BCFF','#4BD29F','#9AFFA2','#F3AC5A','#FF0000']).colors(255);


function init_vis(prototxt){
  // Setup container to hold layer outputs:
  d3.select("#layerLayout")
    .style({
      "margin-left": "210px",
      height: "100%",
      padding:"30px",
      background: "rgba(0,0,0,0.5)",
      display: "none"
    });

  // Setup container to hold tree layout of model layers:
  var treeContainer = d3.select('#treeLayout')
    .style({"margin-left": "210px", height: "100%"})
    .append("div")
      .attr("class", "panel panel-default")
      .style({"height": "100%", overflow: "hidden", background: "#F2F2F2"});

  // Load layers from prototxt file:
  getTreeData('string',prototxt);
  loadTree('#treeLayout .panel',$(treeContainer.node()).height()/2);

  // Add listener for when a layer clicked:
  document.addEventListener("LayerClicked", function(e){
    showLayer(e.layer)
  });

}


function drawModelSelection(selector){
  closePanel();
  var box = d3.select("#treeLayout").node().getBoundingClientRect();
  var formContainer = d3.select(selector).html('')
    .append("div")
      .style({
        "margin-left": "210px",
        height: box.height+"px",
        width: box.width+"px",
        top: box.top+"px",
        padding:"10px 0px",
        background: "rgba(0,0,0,0.5)",
        position: "fixed",
        overflow: "auto"
      }).append("div")
        .attr("class","panel panel-default")
        .style({
          padding: "30px",
          margin: "0 auto",
          top:(box.height - 350)/2 +"px",
          position: "relative",
          height: "350px",
          "width": "400px"});

  // Draw Close Button:
  formContainer.append("div")
    .attr("class", "panel btn btn-sm panel-default")
    .style({
      top: "-25px",
      left: "20px",
      position: "relative",
      float: "right"
    })
    .on("click",function(){
      d3.select(selector).html('');
    })
    .append("span")
      .attr("class", "glyphicon glyphicon-remove");

  // Draw Header:
  formContainer.append("h4").html("Upload Pretrained Model:");

  // Draw Form Items
  var formItems = [{name: "prototxt", ext: ".prototxt", type: "file"},
                   {name: "weights", ext: ".caffemodel", type: "file"},
                   {name: "jobname", ext: "", type: "text"}];

  _.each(formItems, function(item){
    var formGroup = formContainer.append("div").attr("class", "form-group")
    formGroup.append("label")
      .attr("for", item.name)
      .html(item.name);
    formGroup.append("input")
      .attr("class", "form-control")
      .attr({type: item.type, name: item.name, accept: item.ext})
  });

  // Draw Submit Button:
  formContainer.append("button").attr("type", "submit")
    .attr("class", "panel btn panel-default")
    .on("click", function(){
      d3.event.preventDefault();
      $(selector).ajaxSubmit({
        url: window.base_url+"run_model",
        success: function(json){

          var fileInput = $('#image_file');
          fileInput.val('');

          d3.select(selector).style("display","none");
          var treeContainer = d3.select("#treeLayout div").html('');
          getTreeData('string',json.data.prototxt);
          loadTree('#treeLayout .panel',$(treeContainer.node()).height()/2);

          window.load_default = false;
        }
      });
    })
    .html("Upload Model");

  $(selector).ajaxForm();

}




function drawOutputs(container,data){
  // Draws an array of panels representing the layer outputs for a given dataset
  // container -- d3 object representing the container to draw the layer outputs in
  // data      -- a matrix with shape (#outputs,#pixels_per_row,#pixels_per_row, <rgb color Array(3), or greyscale Float> )


  // h, w = dimensions of single output container
  // grid_dim = # pixels per image column
  // pixel_h,w = size of each output pixel
  var h = w = 75;
  var grid_dim = data[0][0].length;
  var pixel_h = pixel_w = h/grid_dim;

  // panel styles:
  var output_style = {margin: "0px",height:h+"px", width:w+"px", position: "relative", cursor: "pointer"};
  var output_width = {height:h, width:w, class: "panel panel-default"};

  var mouseover = {
    height: "85px",
    width: "85px",
    marginTop: "-15px",
    marginLeft: "-10px",
    top: "5px",
    left: "5px",
    zIndex: 1,
    boxShadow: "0px 3px 13px 1px rgba(0,0,0,0.5)"
  };

  var mouseout  = {
    height: "75px",
    width: "75px",
    marginTop: "auto",
    marginLeft: "auto",
    top: "0px",
    left: "0px",
    zIndex: 0,
    boxShadow: "none"
  };


  // iterate through each output
  _.each(data, function(output_data,i){

    // initate a canvas to draw pixels in output on:
    var output_container = container.append("canvas")
      .attr(output_width).style(output_style)
      .on("mouseover", function(d){
        _.extend(this.style,mouseover);
      })
      .on("mouseout",function(){
        _.extend(this.style,mouseout);
      })
      .on("click", function(){
        updateSelected(output_data,i);
      });

    var ctx = output_container.node().getContext("2d");
    drawNeuron(output_data,ctx, pixel_w,pixel_h);

  });

}

function updateSelected(data,neuron){
  var grid_dim = data[0].length;
  var ctx = d3.select("#neuron_visualization").node().getContext("2d");
  drawNeuron(data,ctx, 180/grid_dim,180/grid_dim);
  window.selected_neuron = neuron;
  showDeconv();
}

function drawNeuron(output_data,ctx, pixel_w,pixel_h){
  _.each(output_data,function(row,iy){
    // Iterate through each row:
    _.each(row, function(pixel, ix){
      // Iterate through each column:

      var x   = ix*pixel_w;
      var y   = iy*pixel_h;
      var rgb = "";

      // if rgb array, set color to match values * 255
      if (pixel.length == 3){
        rgb = "rgb("+_.map(pixel,function(n){return Math.floor(n*255)}).join()+")";
      }else {
        // If grayscale , convert to rgb using declared color map
        // use tanh scale, instead of linear scale to match pixel color
        // for better visual appeal:
        // var c = 255*(Math.tanh(2*pixel));
        var c = 255*pixel;
        rgb = window.colormap[Math.floor(c)];
      }

      // draw pixel:
      ctx.fillStyle = rgb;
      ctx.fillRect(x,y,pixel_w,pixel_h);
      ctx.fillRect(x,y,pixel_w,pixel_h);

    });
  });
}




function drawLoadingContainer(selector){
  var box = d3.select("#treeLayout").node().getBoundingClientRect();
  d3.select(selector).html('').style("display","block").append("div")
    .style({
      "margin-left": "210px",
      "text-align": "center",
      height: box.height+"px",
      width: box.width+"px",
      top: box.top+"px",
      padding:"10px 250px",
      background: "rgba(0,0,0,0.5)",
      position: "fixed",
      "z-index":2
    })
    .append("span")
      .attr("class","glyphicon glyphicon-refresh glyphicon-spin");
  return d3.select(selector);
}

function runBackprop() {

  var params = $.param({
    layer_name: window.selected_layer,
    neuron_index: window.selected_neuron,
    path: window.job_path,
    image_key: window.selected_image
  });

  var url = window.base_url+"get_backprop_from_neuron_in_layer?"+params;

  var loadingContainer = drawLoadingContainer("#loadingLayout");

  d3.json(url, function(error, json) {

    loadingContainer.style("display","none");
    var nav = d3.select(".titlebar-nav");

    // Remove old backprops data:
    d3.selectAll(".vis-layer")
      .filter(function(d){return d.type == "backprops"})
      .remove();

    d3.selectAll(".vis-tab")
      .filter(function(d){return d.type == "backprops"})
      .remove();

    // Update titlebar and container of layer layout:
    updateTitlebar([{type: "backprops"}]);
    changeActiveItemInTitlebar("backprops");
    updateLayerData([{data: json.data, type: "backprops"}],"backprops");

    // Draw backprop info:
    updateBackpropInfo(json.info);

  });

}

function showDeconv(){
  var params = $.param({
    layer_name: window.selected_layer,
    neuron_index: window.selected_neuron,
    path: window.job_path,
    image_key: window.selected_image
  });

  var url = window.base_url+"deconv_neuron_in_layer?"+params;

  d3.select("#deconv_container")
    .style({background: "rgba(0,0,0,0.5)"}).html('')
    .append("span")
      .attr("class","glyphicon glyphicon-refresh glyphicon-spin")
      .style({position:"relative", top: "70px"});

  d3.json(url, function(error, json) {
    var container = d3.select("#deconv_container").html('');

    var canvas = container.append("canvas").attr("id", "deconv_visualization");
    container.append("a").attr("class", "btn btn-default btn-sm").html('Lock Output for Backprop?')
      .on("click",runBackprop);
    canvas.attr({height: "180px", width: "180px"});

    var ctx = canvas.node().getContext("2d");
    ctx.clearRect(0, 0, 180,180);
    var grid_dim = json.data[0][0].length;
    drawNeuron(json.data[0],ctx, 180/grid_dim,180/grid_dim);

  });
}


function closePanel(){
  window.selected_layer  = undefined;
  window.selected_type   = undefined;
  window.selected_neuron = undefined;

  d3.select("#layerLayout").style("display", "none").selectAll(".vis-outputs").remove();
}

function showTabInTitlebar(type){
  // Change tab shown when visualizing outputs of a layer:
  _.each(d3.selectAll(".vis-layer")[0], function(node){
    var t = node.dataset.type;
    d3.select(node)
      .style("display", type == t ? "inline-block" : "none");
  });

  changeActiveItemInTitlebar(type);

}

function updateTitlebar(data){
  d3.select(".titlebar-nav").selectAll().data(data)
  .enter()
    .append("div").attr("class",function(d){
      return "panel btn panel-default vis-tab"
    })
    .style({
      "line-height": 1.0,
      width: "90px",
      height: "28px",
      top: "5px",
      position: "relative",
    }).on("click", function(d){
      showTabInTitlebar(d.type)
    })
    .html(function(d){return d.type});
}

function addItemToTitlebar(nav,type){
  nav.append("div")
    .attr("class", "panel btn panel-default"+" vis-tab vis-tab-"+type)
    .style({
      "line-height": 1.0,
      width: "90px",
      height: "28px",
      top: "5px",
      position: "relative",
    })
    .on("click", function(){
      showTabInTitlebar(type)
    })
    .html(type);
}

function changeActiveItemInTitlebar(type){
  window.selected_type = type;
  var activeStyle = {
    background: "#d5ffff",
    cursor: "default",
    border: "1px solid #a7ecec"
  };

  var defaultStyle = {
    background: "#FFFFFF",
    cursor: "pointer",
    border: "1px solid #E6E6E6"
  };

  d3.selectAll(".vis-tab")
    .style(defaultStyle);

  d3.selectAll(".vis-tab")
    .filter(function(d){return d.type == type})
    .style(activeStyle);


}

function buildTitlebar(container, layers, layer){

  // Title Bar:
  var titleBar  = container.append("div")
    .style({
      width: "100%",
      height: "40px",
      "margin-bottom": "2px",
      "text-align": "center",
      "border-bottom":"1px solid #D7D7D7",
      background: "#F3F3F3"}
    );

  // Container Holding Nav Buttons:
  var nav = titleBar.append("span")
    .attr("class", "titlebar-nav")
    .style("display","inline-block");

  // Draw Nav Buttons in Nav Container:
  updateTitlebar(layers);
  changeActiveItemInTitlebar(layer.type);

  // Draw Close Button:
  titleBar.append("div")
    .attr("class", "panel btn btn-sm panel-default")
    .style({
      top: "5px",
      right: "5px",
      position: "relative",
      float: "right"
    })
    .on("click",closePanel)
    .append("span")
      .attr("class", "glyphicon glyphicon-remove");

}


function showLayer(layer, activeTab){
  window.selected_layer = layer.name;

  var params = $.param({
    layer_name: window.selected_layer,
    path: window.job_path,
    image_key: window.selected_image
  });

  var outputs_url  = window.base_url+"get_outputs?"+params;

  var box = d3.select("#treeLayout").node().getBoundingClientRect();

  // Show loading container:
  var loadingContainer = drawLoadingContainer("#loadingLayout");

  d3.json(outputs_url, function(error, json) {
    // Hide loading container
    loadingContainer.html('');

    // Get layers, and set first layer to be initialy viewed:
    var layers = _.filter(json.layers, function(l){return l.data.length >= 1});
    var activeTab = _.isUndefined(activeTab) ? layers[0].type : activeTab;

    // Container Attributes:
    var container_attr  = {class: "panel panel-default vis-outputs"};

    // Container Styles:
    var container_style = {
      margin: "0 auto",
      height: "100%",
      "line-height": "1px",
      "overflow-x": "hidden"
    };

    // Show Layer Layout Above Tree Layout
    var layerContainer  = d3.select("#layerLayout")
      .style({
        display: "block",
        "text-align": "center",
        position: "fixed",
        height: box.height+"px",
        width: box.width+"px",
        top: box.top+"px"
      });

    var titles = _.pluck(layers, "type");

    // Setup container for given outputs:
    var container = layerContainer.append("div")
      .attr(container_attr).style(container_style);

    // Draw titlebar:
    buildTitlebar(container, layers, layers[0]);

    updateLayerData(layers,activeTab);


  });

}

function updateLayerData(layers,activeTab){

  // Draw outputs for each layer type (weights, activations...)
  d3.select(".vis-outputs").selectAll().data(layers).enter()
    .append("div")
      .attr("class","vis-layer")
      .attr("data-type", function(l){return l.type})
      .style("width", "100%")
      .style("display",function(l){
        return l.type == activeTab ? "inline-block" : "none";
      })
      .each(function(l){
        drawOutputs(d3.select(this),l.data);
      });

  // Update which layers to show and hide on existing nodes:s
  d3.selectAll(".vis-layer")
    .style("display",function(l){
      return l.type == activeTab ? "inline-block" : "none";
    });
}

function loadJob(item){
  // Fetch Pretrained Model Weights, and processed images:
  var url = window.base_url + "load_pretrained_model?path=" + item.path;

  d3.json(url, function(error, json) {

    // Set global job path to this items path:
    window.job_path = item.path;
    window.load_default = false;
    window.selected_image = "0";

    // Clear fileInput (incase want to add another image):
    var fileInput = $('#image_file');
    fileInput.val('');

    // Clear Tree Container, and re-draw with new Model Definition:
    var treeContainer = d3.select("#treeLayout div").html('');
    getTreeData('string',json.data.prototxt);
    loadTree('#treeLayout .panel',$(treeContainer.node()).height()/2);

    // Add all images available for visualization to a image carousel
    // TODO: Create a image carousel/handling class:
    var carouselInner = d3.select("#imageCarousel .carousel-inner");
    var canvasAttribtes = {height: 140, width: 140};
    var canvasStyles    = {height: "140px", width: "140px"};

    _.each(json.images, function(image, i){

      // Generate a canvas for each image:
      var canvas = carouselInner.append("canvas")
        .attr(canvasAttribtes)
        .style(canvasStyles)
        .attr("data-key",image.key)
        .attr("class", "item "+ (i == 0 ? "active" : "") );

      // Draw image onto canvas (as its currently saved as array)
      var ctx = canvas.node().getContext("2d");
      ctx.clearRect(0, 0, 140,140);
      var grid_dim = image.img[0][0].length;
      drawNeuron(image.img[0],ctx,140/grid_dim,140/grid_dim);
    });

    console.log(json.data.backprop);
    // Draw info about last backpropagation task
    updateBackpropInfo(json.data.backprop);

  });
}

function changeSelectedImage(){

  if (_.isUndefined(window.selected_layer) || _.isUndefined(window.selected_type)) return;

  var params = $.param({
    layer_name: window.selected_layer,
    image_key: window.selected_image,
    path: window.job_path
  });

  var url = window.base_url+"get_"+window.selected_type+"?"+params;

  d3.json(url, function(error, json){

    d3.selectAll(".vis-layer")
      .filter(function(d){return d.type == window.selected_type})
      .remove();

    updateLayerData([{data: json.data, type: window.selected_type}],window.selected_type);
    updateSelected(json.data[window.selected_neuron],window.selected_neuron);

  });
}

function updateBackpropInfo(info){
  // Draw backprop information into the model visualizations pane:
  var canvas = d3.select("#locked_backprop_container").html('')
    .style("display","block")
    .append("canvas").attr({height: "180px", width: "180px"});

  var ctx = canvas.node().getContext("2d");
  ctx.clearRect(0, 0, 180,180);
  var grid_dim = info.data[0].length;

  drawNeuron(info.data,ctx, 180/grid_dim,180/grid_dim);

  d3.select("#backprop_layer").html("<b>Backprop Layer: "+info.attrs.layer+" </b>");
  d3.select("#backprop_output").html("<b>Output Number: "+info.attrs.neuron+" </b>");
}
