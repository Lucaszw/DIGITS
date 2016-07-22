var colormap = chroma.scale(['#541E8A','#3F84FE','#87BCFF','#4BD29F','#9AFFA2','#F3AC5A','#FF0000']).colors(255);

var LayerVisualizations = function(selector,props){
  var self = this;

  self.extend = function(props){
      props = _.isUndefined(props) ? {} : props;
      return _.extend(props, {parent: self});
  };

  self.actions  = new LayerVisualizations.Actions(self.extend(props));
  self.tree_container = d3.select(selector);

  self.carousel = null;
  self.panel    = null;
  self.overlay  = null;
  self.jobs     = null;
  self.job_id   = null;
  self.image_id = null;
  self.layer    = null;
  self.range    = null;
  self.outputs  = [];

  self.initPanel = function(props){
    var selector = self.tree_container.node();
    self.overlay = new LayerVisualizations.Overlay(selector,self.extend(props));
    self.panel   = new LayerVisualizations.Panel(selector,self.extend(props));
  };

  self.initCarousel = function(selector,props){
    self.carousel = new LayerVisualizations.Carousel(selector,self.extend(props));
  };

  self.initJobs = function(selector,props){
    self.jobs = new LayerVisualizations.Jobs(selector,self.extend(props));
    self.jobs.render();
  };

  self.dispatchInference = function() {
    if (!_.isNull(self.layer)){
      self.actions.getInference(self.layer.name);
    }
  };

  self.update = function(){

    var h = w = 75;
    var grid_dim = self.outputs[0][0].length;
    var pixel_h = pixel_w = h/grid_dim;

    var output_style = {margin: "0px",height:h+"px", width:w+"px", position: "relative"};
    var output_attr  = {height:h, width:w, class: "panel panel-default"};

    var items = self.panel.body.selectAll("canvas").data(self.outputs);
    items.attr("class", "item").enter()
      .append("canvas")
        .attr(output_attr).style(output_style)
        .each(function(data,i){
          var ctx = this.getContext("2d");
          ctx.clearRect(0, 0, w, h);
          self.drawUnit(data,ctx,pixel_w,pixel_h);
        });

    items.exit().remove();

  };


  self.drawOutputs = function(json){
    self.panel.render();
    self.outputs.length = 0;
    self.outputs.push.apply(self.outputs, _.isUndefined(json.data) ? [] : json.data);
    self.update();
    self.panel.drawNav(self.range, json.length);
  };

  self.drawUnit = function(output_data,ctx, pixel_w,pixel_h){
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
  };

  // Add listener for when a layer clicked:
  document.addEventListener("LayerClicked", function(e){
    self.layer = e.layer;
    self.range = {min: 0 , max: 40};
    self.dispatchInference();
  });

};

LayerVisualizations.Actions = function(props){
  var self   = this;
  var parent = !_.isUndefined(props.parent) ? props.parent : props;

  self.getInference = function(layerName){

    params = $.param({
      "job_id": parent.job_id,
      "image_id":parent.image_id,
      "layer_name":layerName,
      "range_min": parent.range.min,
      "range_max": parent.range.max
    });

    parent.overlay.render();

    var outputs_url  = "/pretrained_models/get_inference.json?"+params;
    d3.json(outputs_url, function(error, json) {
      parent.drawOutputs(json);
    });
  };

  self.getOutputs = function(job_id){
    parent.overlay.render();
    var outputs_url  = "/pretrained_models/get_outputs.json?job_id="+job_id;
    d3.json(outputs_url, function(error, json) {
      parent.overlay.remove();
      parent.job_id = job_id;
      parent.jobs.load(json);
    });
  };

  self.uploadImage = function(file){
     parent.overlay.render();
     var upload_url = "/pretrained_models/upload_image.json?job_id="+parent.job_id;
     var formData = new FormData();
     // Check file type.
     if (!file.type.match('image.*')) {
       console.error("Bad File Type");
       return;
     }
     // Add the file to the request.
     formData.append('image', file, file.name);
     var xhr = new XMLHttpRequest();
     xhr.onload = function () {
        parent.overlay.remove();
        if (xhr.status === 200) {
          var json = JSON.parse(xhr.responseText);
          parent.carousel.load(json);
        } else {
          console.error("Failed to Upload File");
        }
     };
    // Send Request:
    xhr.open("POST", upload_url, true);
    xhr.send(formData);
  };

};

LayerVisualizations.Panel = function(selector,props){
  var self = this;
  var parent = !_.isUndefined(props.parent) ? props.parent : props;

  self.container = d3.select(selector);
  self.outer = null;
  self.headingCenter = null;
  self.headingRight = null;
  self.body  = null;
  self.nav   = null;

  self.drawCloseButton = function(){
    self.headingRight.append("a")
      .attr("class", "btn btn-xs btn-danger")
      .on("click",self.remove)
      .append("span").attr("class", "glyphicon glyphicon-remove");
  };

  self.updateOuputs = function(d,step){
    parent.range = {min: d*step, max: (d+1)*step};
    parent.dispatchInference();
  };

  self.drawNav = function(range,n){
    var step = range.max - range.min;
    var ul   = self.nav.html("").append("nav")
                .append("ul").attr("class", "pagination");

    var numSteps   = Math.ceil(n/step);

    ul.selectAll("li").data(_.range(0,numSteps)).enter()
      .append("li")
        .attr("class", function(d){ return (d*step == range.min) ? "active" : ""})
        .style("cursor","pointer")
        .append("a")
          .html(function(d){ return d*step+"-"+(d+1)*step})
          .on("click", function(d){
            self.updateOuputs(d,step);
          });
  };

  self.remove = function(){
    parent.layer = null;
    parent.overlay.remove();
  };

  self.render = function(){
    self.container.style("position","relative");

    self.outer = parent.overlay.inner.append("div").attr("class", "component-outer");
    self.outer.style("padding", "30px");

    var panel   = self.outer.append("div").attr("class", "panel panel-default");
    var heading = panel.append("div").attr("class", "panel-heading")
      .append("div").attr("class","row").style("padding","0px 10px");

    self.headingCenter = heading.append("div").attr("class", "text-center col-xs-offset-1 col-xs-10");
    self.headingRight  = heading.append("div").attr("class", "col-xs-1 text-right");

    var panelBody = panel.append("div").attr("class", "panel-body");
    self.body = panelBody.append("div");
    self.nav = panelBody.append("div");

    self.drawCloseButton();
  };

};


LayerVisualizations.Overlay = function(selector,props){
  var self   = this;
  var props  = _.isUndefined(props) ? {} : props;
  var parent = !_.isUndefined(props.parent) ? props.parent : props;

  self.container = d3.select(selector);
  self.inner = null;

  self.render = function(){
    self.remove();
    self.container.style("position","relative");
    self.inner = self.container.append("div")
      .attr("class", "text-center component-outer loading-overlay");

    self.inner.style("background", "rgba(0,0,0,0.5)");

    var spinner = self.inner.append("span")
      .attr("class", "glyphicon glyphicon-refresh glyphicon-spin");

    spinner.style({top: "50%", "margin-top": "-20px", color: "white", "font-size": "40px"})

    return self.inner;
  };

  self.remove = function(){
    d3.selectAll(".loading-overlay").remove();
  };

};

LayerVisualizations.Carousel = function(selector,props){
  var self   = this;
  var parent = !_.isUndefined(props.parent) ? props.parent : props;

  self.container = d3.select(selector);

  self.images     = new Array();
  self.inner      = null;
  self.fileSelect = null;

  self.dispatchUpload = function(e){
    var file = self.fileSelect.node().files[0];
    parent.actions.uploadImage(file);
  };
  self.dispatchChangeImage = function(e){
    parent.image_id = e.relatedTarget.dataset.imageId;
    parent.dispatchInference();
  };

  self.update = function(){
    var bbox = self.inner.node().getBoundingClientRect();
    var size = {height: bbox.height, width: bbox.width};

    var items = self.inner.selectAll("canvas").data(self.images);
    var n = self.images.length-1;
    items.attr("class", "item").enter()
      .append("canvas")
        .attr("class",function(d,i){return "item "+(i == n ? "active" : "")})
        .attr(size)
        .attr("data-image-id",function(d){return d.id})
        .style({height: size.height+"px", width: size.width+"px"})
        .each(function(image,i){
           var ctx = this.getContext("2d");
           ctx.clearRect(0, 0, size.width,size.height);
           var grid_dim = image.data[0].length;
           parent.drawUnit(image.data,ctx,size.width/grid_dim,size.height/grid_dim);
        });

    items.exit().remove();

    parent.image_id = self.images[n].id;
  };

  self.load = function(image){
    self.images.push(image);
    parent.image_id = image.id;
    self.update();
  };

  self.drawButton = function(){
    var button = self.container.append("div")
      .attr("class","file-upload btn btn-primary")
      .style({position: "relative", display: "block"});

    button.append("span").text("Upload Image");

    self.fileSelect = button.append("input").attr({
      type: "file",
      class: "upload",
      multiple: ""
    });

    self.fileSelect.node().onchange = self.dispatchUpload;

  };

  self.drawCarousel = function(){
    var outer = self.container.append("div")
      .attr("id", "imageCarousel")
      .attr("class","panel panel-default  carousel slide")
      .style(self.styles.outer);

    self.inner = outer.append("div").attr("class","carousel-inner").style("height","100%");

    outer.append("a").attr(self.attr.left)
      .append("span").attr("class","glyphicon glyphicon-chevron-left");

    outer.append("a").attr(self.attr.right)
      .append("span").attr("class","glyphicon glyphicon-chevron-right");

    $("#imageCarousel").carousel({ interval: false });

    $('#imageCarousel').bind('slide.bs.carousel', self.dispatchChangeImage);
  };

  self.remove = function(){
    self.container.html('');
  }

  self.render = function(images){
    self.container.attr("class","text-center");
    self.container.append("b").text("Saved Images:");
    self.drawCarousel();
    self.drawButton();

    self.images.length = 0;
    self.images.push.apply(self.images, _.isUndefined(images) ? [] : images);
    self.update();

  };

  self.attr = {
    left: {
      "class": "left carousel-control",
      "href": "#imageCarousel",
      "role": "button",
      "data-slide": "prev"
    },
    right: {
      "class": "right carousel-control",
      "href": "#imageCarousel",
      "role": "button",
      "data-slide": "next"
    }
  };

  self.styles = {
    outer: {
      margin: "0 auto 10px auto",
      width: "100%",
      height: "140px",
      overflow: "hidden"
    }
  };

};

LayerVisualizations.Jobs = function(selector,props){
  var self   = this;
  var parent = !_.isUndefined(props.parent) ? props.parent : props;

  self.jobs      = props.jobs;
  self.container = d3.select(selector);
  self.tree      = null;
  self.layers    = null;

  self.update = function(){
    var items = self.container.selectAll("div").data(self.jobs);
    items.enter()
      .append("div")
        .attr("class","btn btn-xs btn-default")
        .style(LayerVisualizations.Styles.button);

    items.append("div").html(function(d){return d.name});
    items.append("div").attr("class","subtle").html(function(d){return d.id});
    items.on("click",self.dispatch);

    items.exit().remove();
  };

  self.dispatch = function(d){
    parent.carousel.remove();
    parent.actions.getOutputs(d.id);
  }

  self.load = function(json){
    console.log(json.framework);
    console.log(parent.tree_container.node());
    if (json.framework == "caffe"){
      var d = getTreeData("text",json.model_def);
      self.layers = d.layers;
      self.tree   = d.tree;
      loadTree(parent.tree_container.node());
    } else {
      generateTorchTree(json.model_def);
      loadTorchTree(parent.tree_container.node());
    }
    parent.carousel.render(json.images);
  }

  self.render = function(d){
    self.container.style(self.styles.jobs);
    self.update();
  };

  self.styles = {
    jobs: {
      padding: "5px",
      "max-height": "200px",
      "overflow-y": "scroll",
      width: "204px",
      left: "-4px",
      position: "relative"
    }
  }

};

LayerVisualizations.Styles  = {
  button: {
    "margin-bottom": "1px",
    background: "white",
    width: "100%",
    "box-shadow": "none"
  }
}
