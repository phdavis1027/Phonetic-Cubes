const fs = require('fs')
const path = require('path')
const Viva = require('../lib/VivaGraphJS/dist/vivagraph')


graphTools = {}
graphTools.fetchGraphData = function(){
    let data = JSON.parse(fs.readFileSync(path.resolve("public/json/cube.json")))
    return data 
}

graphTools.constructGraph = function(data){

    let graph = Viva.Graph.graph()


    for(let i = 0; i < data.length; i++){
        let currentNode = String(data[i].id)
        let adjacencies = data[i].adjacencies
        for(let j = 0; j < adjacencies.length; j++){
            let neighbor = String(adjacencies[j].nodeTo)
            graph.addLink(currentNode, neighbor)
        }
    }
/* This code was my "just get it to work" approach and I'm keeping it around in case something breaks

    for(let i = 0; i < data.length; i++){
        let currentNode = String(data[i].id)
        let adjacencies = data[i].adjacencies
        for(let j = 0; j < adjacencies.length; j++){
            graph.addLink(currentNode, String(adjacencies[j].nodeTo))
        }
    }
*/
    return graph;
}

graphTools.setupRenderer = function(graph){
    var layout = Viva.Graph.Layout.forceDirected(graph, {
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

module.exports = graphTools;