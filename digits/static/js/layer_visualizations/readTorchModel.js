function generateTorchTree(nodes){
  nodes = JSON.parse(nodes);
  function getLeafNodes(leafNodes, obj){
    // Get the leaf nodes of a given branch
     if(obj.isLast != true){
         obj.children.forEach(function(child){getLeafNodes(leafNodes,child)});
     } else{
         leafNodes.push(obj);
     }
  }

  function getNodesAndLinks(nodes,links,obj){
    // Get the links between each node
    var source  = obj;
    if(obj.isLast != true){
      obj.children.forEach(function(child){
        links.push({source: source, target: child});

        if (_.pluck(nodes,"index").indexOf(child.index) != -1) return;
        nodes.push(child);
        getNodesAndLinks(nodes,links,child);

      });
    }
  }

  function toGraph(node){
    // Convert Tree Structure to Graph Structure
    var links = new Array();
    var nodes = new Array();
    var graph = {nodes: new Array(), links: new Array()};

    graph.nodes.push(node);
    getNodesAndLinks(graph.nodes, links, node);

    graph.links = links;

    return graph;
  }

  function isContainer(nodeType){
    return (nodeType == "nn.Sequential" || nodeType == "nn.Concat" || nodeType == "nn.Parallel" || nodeType == "nn.DepthConcat")
  }
  function isParallel(nodeType){
    return (nodeType == "nn.Concat" || nodeType == "nn.Parallel" || nodeType == "nn.DepthConcat")
  }

  function chainContents(node){
    // Make the parent of each sibling its previous sibling

    node.children = [node.contents[0]];
    node.contents[0].parents = [node];
    var prevNode = node;

    _.each(node.contents, function(child,i){

      child.parents = [prevNode];

      prevNode = child;

      if (isContainer(child.type)) {
        var exit = isParallel(child.type) ? branchContents(child) : chainContents(child);
        prevNode = child = exit;
        exit.isLast = false;
      }

      if (i != node.contents.length-1) child.children = [node.contents[i+1]];
      if (i == node.contents.length-1) child.isLast   = true;
    });
    var exitIndex = prevNode.index + 0.5;
    var exit = {index: exitIndex, type: "s-exit", children: [], parents: [], isLast: true};
    prevNode.children = [exit];
    prevNode.isLast = false;
    exit.parents = [prevNode];
    return exit;

  }

  function branchContents(node){
    // Create branch structure, that terminates with the leaves
    // of each branch joining together

    var leafNodes = new Array();
    node.children = new Array();
    _.each(node.contents, function(child,i){
      if (child.type == "nn.Sequential") chainContents(child);
      child.parents = [node];
      node.children.push(child);
    });

    getLeafNodes(leafNodes,node);
    var exitIndex = _.max(_.pluck(leafNodes, "index")) + 0.4;
    var exit = {index: exitIndex, type: "Concat", children: [], parents: [], isLast: true}
    _.each(leafNodes, function(leaf){
      leaf.children = [exit];
      leaf.isLast   = false;
      exit.parents.push(leaf);
    });

    return exit;
  }

  function getContainerContents(node){
    return _.filter(nodes, function(n){return n.container.index == node.index})
  }

  _.each(nodes, function(node){
    node.contents = getContainerContents(node);
    // if (node.type == "nn.Sequential") chainContents(node);
  });

  chainContents(nodes[0]);
  graph = toGraph(nodes[0]);

}
