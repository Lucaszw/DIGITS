// d3 alert menu for uploading a pretrained model:
// Call renderer to draw component:

var PretrainedModel = function(props){

  var self = this;
  self.container = d3.select(props.selector);
  self.innerContainer = null;
  self.frameworkSelector = null;

  self.frameworkChanged = function(){
    var nextFramework = self.frameworkSelector.property("value");
    if (nextFramework ==  "torch"){
      self.renderTorchForm();
    }else{
      self.renderCaffeForm();
    }
  };

  self.render = function(){
    self.container.html('');
    var row = self.container.append("div").attr("class","row");
    self.input(row.append("div").attr("class","col-xs-6"),"text","Jobname", "job_name");

    self.frameworkSelector = self.select(
      row.append("div").attr("class","col-xs-6"),["caffe", "torch"],"Framework","framework"
    );
    self.frameworkSelector.on("change", self.frameworkChanged);
    self.innerContainer = self.container.append("div");
    self.renderCaffeForm();
  };

  self.renderCaffeForm = function(){
    self.innerContainer.html('');

    self.input(self.innerContainer,"file","Weights (**.caffemodel)", "weights_file");
    self.input(self.innerContainer,"file","Model Definition: (original.prototxt)", "model_def_file");
    self.input(self.innerContainer,"file","Labels file: (Optional)", "labels_file");

    self.innerContainer.append("button").attr({type: "submit",class: "btn btn-default"})
      .on("click",self.submit)
      .style("background","white")
      .html("Upload Model");
  };

  self.renderTorchForm = function(e){
    self.innerContainer.html('');

    self.input(self.innerContainer,"file","Weights (**.t7)", "weights_file");
    self.input(self.innerContainer,"file","Model Definition: (model.lua)", "model_def_file");
    self.input(self.innerContainer,"file","Labels file: (Optional)", "labels_file");

    self.innerContainer.append("button").attr({type: "submit",class: "btn btn-default"})
      .on("click",self.submit)
      .style("background","white")
      .html("Upload Model");
  };


  self.input = function(obj,type,label,name){
    var group = obj.append("div").attr("class","form-group");
    group.append("label")
        .attr("for",name).html(label);
    group.append("input")
      .attr({type: type, class: "form-control", name: name});
  };

  self.select = function(obj,data,label,name){
    var group = obj.append("div").attr("class","form-group");
    group.append("label")
        .attr("for",name).html(label);

    var mySelect = group.append("select").attr({
        class: "form-control",
        name: name
      });

    mySelect.selectAll('option').data(data).enter()
      .append("option")
        .attr("value",function(d){return d})
        .text(function(d){return d});

    return mySelect;
  }

};
