function setupRenderer(graph){    var layout = Viva.Graph.Layout.forceDirected(graph, {
        springLength : 10,
        springCoeff : 0.0008,
        dragCoeff : 0.02,
        gravity : -1.2
    });

    var graphics = Viva.Graph.View.svgGraphics();

    var renderer = Viva.Graph.View.renderer(graph, {
            layout     : layout,
            graphics   : graphics
        });

    return renderer;
}


function constructGraph(data, graph){
    for(let i = 0; i < data.length; i++){
        let currentNode = String(data[i].id)
        let adjacencies = data[i].adjacencies
        for(let j = 0; j < adjacencies.length; j++){
            let neighbor = String(adjacencies[j].nodeTo)
            graph.addLink(currentNode, neighbor)
        }
        graph.getNode(currentNode).data = {active: data[i].data['active']}
    }
    
}