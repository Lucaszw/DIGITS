// d3 alert menu for uploading a pretrained model:
// Call renderer to draw component:

var PretrainedModel = function(props){

  var self = this;
  var inputs = PretrainedModel.mixins;

  self.container = d3.select(props.selector);
  self.innerContainer = null;
  self.frameworkSelector = null;
  self.frameworks = [
    {text: "Caffe", value: "caffe"},
    {text: "Torch", value: "torch"}
  ];
  self.resize_channels = [
    {text: "Color", value: 3},
    {text: "Grayscale", value: 1}
  ];

  self.resize_modes = [
    {text:"Squash", value: "squash"},
    {text: "Crop", value: "crop"},
    {text: "Fill", value: "fill"},
    {text: "Half Crop, Half Fill", value: "half_crop"}
  ];

  self.frameworkChanged = function(){
    var nextFramework = self.frameworkSelector.property("value");
    if (nextFramework ==  "torch"){
      self.renderTorchForm();
    }else{
      self.renderCaffeForm();
    }
  };
  self.newRow = function(){
    return self.container.append("div").attr("class","row");
  };

  self.render = function(){
    self.container.html('');
    var row = self.newRow();
    inputs.field(row.append("div").attr("class","col-xs-6"),"text","Jobname", "job_name");

    self.frameworkSelector = inputs.select(
      row.append("div").attr("class","col-xs-6"),self.frameworks,"Framework","framework"
    );

    row = self.newRow();
    row.style("background", "whitesmoke");
    row.style("border-radius", "5px 5px 0px 0px");
    inputs.select(
      row.append("div").attr("class","col-xs-6"),
        self.resize_channels,"Image Type","image_type"
    );

    inputs.select(
      row.append("div").attr("class","col-xs-6"),
        self.resize_modes,"Resize Mode","resize_mode"
    );

    row = self.newRow();
    row.style("background", "whitesmoke");
    row.style("border-radius", "0px 0px 5px 5px");

    inputs.field(row.append("div").attr("class","col-xs-6"),"number","Width", "width").attr("value",256);
    inputs.field(row.append("div").attr("class","col-xs-6"),"number","Height", "height").attr("value",256);

    self.frameworkSelector.on("change", self.frameworkChanged);
    self.innerContainer = self.container.append("div");
    self.renderCaffeForm();
  };

  self.renderCaffeForm = function(){
    self.innerContainer.html('');

    inputs.field(self.innerContainer,"file","Weights (**.caffemodel)", "weights_file");
    inputs.field(self.innerContainer,"file","Model Definition: (original.prototxt)", "model_def_file");
    inputs.field(self.innerContainer,"file","Labels file: (Optional)", "labels_file");

    self.innerContainer.append("button").attr({type: "submit",class: "btn btn-default"})
      .on("click",self.submit)
      .style("background","white")
      .html("Upload Model");
  };

  self.renderTorchForm = function(e){
    self.innerContainer.html('');

    inputs.field(self.innerContainer,"file","Weights (**.t7)", "weights_file");
    inputs.field(self.innerContainer,"file","Model Definition: (model.lua)", "model_def_file");
    inputs.field(self.innerContainer,"file","Labels file: (Optional)", "labels_file");

    self.innerContainer.append("button").attr({type: "submit",class: "btn btn-default"})
      .on("click",self.submit)
      .style("background","white")
      .html("Upload Model");
  };

};

PretrainedModel.mixins = {
  select: function(obj,data,label,name){
    var group = obj.append("div").attr("class","form-group");
    group.append("label")
        .attr("for",name).html(label);

    var mySelect = group.append("select").attr({
        class: "form-control",
        name: name
      });

    mySelect.selectAll('option').data(data).enter()
      .append("option")
        .attr("value",function(d){return d.value})
        .text(function(d){return d.text});

    return mySelect;

  },
  field: function(obj,type,label,name){
    var group = obj.append("div").attr("class","form-group");
    group.append("label")
        .attr("for",name).html(label);
    return group.append("input")
      .attr({type: type, class: "form-control", name: name});
  }

};
