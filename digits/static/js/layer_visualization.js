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

  // Setup model selector container:
  drawModelSelection('#selectModelLayout');


  // Load layers from prototxt file:
  getTreeData('string',prototxt);
  loadTree('#treeLayout .panel',$(treeContainer.node()).height()/2);

  // Add listener for when a layer clicked:
  document.addEventListener("LayerClicked", function(e){
    showLayer(e.layer)
  });

}

function drawModelSelection(selector){
  var formContainer = d3.select(selector)
    .style({
      "margin-left": "210px",
      height: "100%",
      padding:"250px",
      background: "rgba(0,0,0,0.5)",
      position: "relative",
      display: "none",
      top: -1*d3.select("#treeLayout").node().getBoundingClientRect().height + 'px'
    }).append("div")
      .attr("class","panel panel-default")
      .style({height: "100%", padding: "30px"});

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
      d3.select(selector).style("display","none");
    })
    .append("span")
      .attr("class", "glyphicon glyphicon-remove");

  // Draw Header:
  formContainer.append("h4").html("Upload Pretrained Model:");

  // Draw Form Items
  var formItems = [{name: "prototxt", ext: ".prototxt"}, {name: "weights", ext: ".caffemodel"}, {name: "mean", ext: ".npy"}];

  _.each(formItems, function(item){
    var formGroup = formContainer.append("div").attr("class", "form-group")
    formGroup.append("label")
      .attr("for", item.name)
      .html(item.name);
    formGroup.append("input")
      .attr({type: "file", name: item.name, accept: item.ext})
  });

  // Draw Submit Button:
  formContainer.append("button").attr("type", "submit")
    .attr("class", "panel btn panel-default")
    .on("click", function(){
      d3.event.preventDefault();
      $(selector).ajaxSubmit({
        url: window.base_url+"run_model",
        success: function(json){
          console.log(json);
          console.log("Model is ready to go!");
        }
      });
    })
    .html("Upload Model");

  $(selector).ajaxForm();

}

function drawKernel(container,data){
  // Draws an array of panels representing the layer outputs for a given dataset
  // container -- d3 object representing the container to draw the layer outputs in
  // data      -- a matrix with shape (#outputs,#pixels_per_row,#pixels_per_row, <rgb color Array(3), or greyscale Float> )

  // Default to faded red,yellow,green,blue as color map (only applies if greyscale data):
  var colormap = chroma.scale(['#541E8A','#3F84FE','#87BCFF','#4BD29F','#9AFFA2','#F3AC5A','#FF0000']).colors(255);

  // h, w = dimensions of single output container
  // grid_dim = # pixels per image column
  // pixel_h,w = size of each output pixel
  var h = w = 50;
  var grid_dim = data[0][0].length;
  var pixel_h = pixel_w = h/grid_dim;

  // panel styles:
  var output_style = {margin: "0px",height:h+"px", width:w+"px"};
  var output_width = {height:h, width:w, class: "panel panel-default"};


  // iterate through each output
  _.each(data, function(output_data,i){

    // draw a maximum of 1000 outputs (for better performance)
    if (i > 1000) return;

    // initate a canvas to draw pixels in output on:
    var output_container = container.append("canvas")
      .attr(output_width).style(output_style);
    var ctx = output_container.node().getContext("2d");

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
          var c = 255*(Math.tanh(2*pixel));
          rgb = colormap[Math.floor(c)];
        }

        // draw pixel:
        ctx.fillStyle = rgb;
        ctx.fillRect(x,y,pixel_w,pixel_h);

      });
    });

  });

}

function closePanel(){
  d3.select("#layerLayout").style("display", "none").html('');
}

function showTab(vis_type){
  // Change tab shown when visualizing outputs of a layer:
  _.each(d3.selectAll(".vis-layer")[0], function(node){
    var type = node.dataset.type;
    d3.select(node)
      .style("display", type == vis_type ? "block" : "none");
  });
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
    .style("display","inline-block");

  // Draw Nav Buttons in Nav Container:
  _.each(layers, function(l){

    var border_style = "1px solid " + (layer.vis_type == l.vis_type ? "#a7ecec" : "#E6E6E6");
    var cursor_style = layer.vis_type == l.vis_type ? "default" : "pointer";
    var bg_style    = layer.vis_type == l.vis_type ? "#d5ffff" : "#FFFFFF";
    nav.append("div")
      .attr("class", "panel btn panel-default")
      .style({
        "line-height": 1.0,
        width: "90px",
        height: "28px",
        top: "5px",
        position: "relative",
      })
      .on("click", function(){showTab(l.vis_type)})
      .style("cursor", cursor_style)
      .style("background", bg_style)
      .style("border", border_style)
      .html(l.vis_type)
  });


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

function showLayer(layer){
  // ** Requires window.outputs_data variable container outputs for each layer **

  // Get outputs of layer that was clicked:
  var selected_layers = _.filter(window.outputs_data,function(d){return d.name == layer.name});

  // If clicked layer has no outputs then exit
  if (selected_layers.length < 1) return;

  // Container Styles:
  var container_attr = {class: "panel panel-default vis-layer"};
  var container_style = {height: "100%", overflow: "auto", "line-height": "1px"};
  var layerContainer = d3.select("#layerLayout").style("display","block").html('');

  // Draw outputs inside the layerLayout div container:
  _.each(selected_layers, function(outputs,i){

    // Setup container for given outputs
    var container = layerContainer.append("div")
      .attr(container_attr).style(container_style)
      .attr("data-type", outputs.vis_type);

    // Hide all but first set of outputs at first:
    container.style("display", i == 0 ? "block" : "none");

    // Draw titlebar:
    buildTitlebar(container, selected_layers,outputs);

    // Draw outputs:
    drawKernel(container, outputs.data);
  });

  // Show container holding all outputs for this layer:
  layerContainer.style({
    position: "relative",
    top: -1*d3.select("#treeLayout").node().getBoundingClientRect().height + 'px'
  });

}
