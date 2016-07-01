$(document).ready(function(){
  d3.select("body").style("background","#F9F9F9");

  $( ".container .col" ).wrapInner( "<div class='col-content'></div>");
  $( ".container .col-content" ).wrap( "<div class='col-wrapper'></div>");

  d3.selectAll(".col:not(.col-nopanel) .col-wrapper")
    .classed("panel panel-default text-center", true)
    .style("padding", "15px");

  d3.selectAll(".col-big .col-wrapper").style("height","200px");
  d3.selectAll(".col-tall .col-wrapper").style("height","480px");
  d3.selectAll(".col-norm .col-wrapper").style("height","120px");
  d3.selectAll(".col-small .col-wrapper").style("height","30px");

  d3.selectAll(".col-status .col-wrapper").style({
    background: "rgb(164, 230, 200)",
    border: "1px solid rgb(124, 171, 157)"
  });

  centerContent();

});

function centerContent(selector){
  var grid_contents = d3.selectAll(".container .center .col-content")[0];

  _.each(grid_contents, function(container){
    // Compute the height of each container
    var bbox = container.getBoundingClientRect();

    // Re-position container to be centered based on its height
    d3.select(container).style({
        position: "relative",
        top: "50%",
        "margin-top": -1*bbox.height/2 +"px"
    });
  });
}
