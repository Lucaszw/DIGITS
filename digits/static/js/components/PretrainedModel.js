// d3 alert menu for uploading a pretrained model:
// Set url to tell form where to submit & and call renderer to draw component:

var PretrainedModel = function(props){
  var self = this;
  var _url = $SCRIPT_ROOT+"/models/images/visualizations/new";

  self.size      = 350;
  self.url       = _.isUndefined(props.url) ? _url : props.url;
  self.container = null;
  self.form      = null;

  self.input = function(obj,type,label,name){
    var group = obj.append("div").attr("class","form-group");
    group.append("label")
        .attr("for",name).html(label);
    group.append("input")
      .attr({type: type, class: "form-control", name: name});
  };

  self.close = function(){
    self.container.html('');
  };

  self.render = function(e){
    e.preventDefault();

    self.container = d3.select("body")
    .append("div").html('');

    var outer = self.container.append("div").style(self.styles.outer).on("click",self.close);

    // Draw form body:
    var body = outer.append("div")
      .attr("class", "panel panel-default")
      .style(self.styles.body)
      .on("click",function(){d3.event.stopPropagation()});

    // Draw close button:
    body.append("div")
      .style({width:"100%", "text-align": "right"})
      .append("a")
        .attr("class","btn btn-default btn-xs")
        .style(self.styles.closeButton)
        .on("click", self.close)
        .append("span")
          .attr("class","glyphicon glyphicon-remove");

    // Draw Form:
    self.form = body.append("form")
      .attr({action: self.url, method: "post", enctype: "multipart/form-data"})
      .style("padding", "10px "+self.size/10+"px");

    self.form.append("h4").style(self.styles.h4).html("Upload Pretrained Model");

    self.input(self.form,"file","deploy.prototxt", "prototxt_file");
    self.input(self.form,"file","***.caffemodel", "caffemodel_file");
    self.input(self.form,"text","Jobname", "job_name");

    self.form.append("button").attr({type: "submit",class: "btn btn-default"})
      .on("click",self.submit)
      .style("background","white")
      .html("Upload Model");

  };

  self.styles = {
    closeButton: {
      background: "#ff7e7e",
      color: "white",
      border: "1px solid #d42525"
    },
    body: {
      height: self.size + "px",
      width: self.size + "px",
      top: "50%",
      margin: -self.size/2 + "px auto",
      position: "relative"
    },
    h4: {
      position: "relative",
      top: "-10px",
      margin: "0px"
    },
    outer: {
      width: "100%",
      height: "100%",
      background: "rgba(0,0,0,0.5)",
      position: "fixed",
      top: "0px",
      "z-index": 2
    }
  };

};
