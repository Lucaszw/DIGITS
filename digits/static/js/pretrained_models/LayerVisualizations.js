var LayerVisualizations = function(props){
  var self = this;
};

LayerVisualizations.Carousel = function(selector,props){
  var self = this;
  self.container = d3.select(selector);
  self.inner   = null;
  self.enabled = false;

  self.dispatchUpload = function(){
    console.log("Todo: Assign Me an Action!");
  }

  self.drawButton = function(){
    var btn = self.container.append("a")
      .attr("class","btn btn-sm btn-primary")
      .style(LayerVisualizations.Styles.button)
      .on("click",self.dispatchUpload)
      .text("Upload Image to Job");
  };

  self.drawCarousel = function(){
    var outer = self.container.append("div")
      .attr("id", "imageCarousel")
      .attr("class","panel panel-default  carousel slide")
      .style(self.styles.outer);

    self.inner = outer.append("div").attr("class","carousel-inner");

    outer.append("a").attr(self.attr.left)
      .append("span").attr("class","glyphicon glyphicon-chevron-left");

    outer.append("a").attr(self.attr.right)
      .append("span").attr("class","glyphicon glyphicon-chevron-right");

    $("#imageCarousel").carousel({ interval: false });

    $('#imageCarousel').bind('slide.bs.carousel', function (e) {
      window.selected_image = e.relatedTarget.dataset.key;
      changeSelectedImage();
    });
  };

  self.render = function(){
    self.container.attr("class","text-center").style("opacity","0.5");
    self.drawCarousel();
    self.drawButton();
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

}

LayerVisualizations.Jobs = function(selector,props){
  var self = this;

  self.jobs      = props.jobs;
  self.actions   = props.actions;
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
    self.actions.getOutputs(self,d.id);
  }

  self.load = function(d){
    var d = getTreeData("text",d.model_def);
    self.layers = d.layers;
    self.tree   = d.tree;
    loadTree("#treeContainer")
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
    width: "100%"
  }
}

LayerVisualizations.Actions = function(props){
  self = this;

  self.getOutputs = function(Jobs,job_id){
    var outputs_url  = "/pretrained_models/get_outputs.json?job_id="+job_id;
    d3.json(outputs_url, function(error, json) {
      Jobs.load(json);
    });
  }
};
