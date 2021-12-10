var express = require('express');
const { appendFile } = require('fs');
var router = express.Router();
const fs = require('fs')
const path = require('path')
const graphTools = require('../data/graphTools');

/* GET home page. */

router.use('/', (req, res, next)=>{
  let data = graphTools.fetchGraphData();
  res.locals.graphFootball = data
  next()
})

/*
router.use('/',(req, res, next)=>{
  let data = res.locals.graphFootball;
  let graph = graphTools.constructGraph(data)
  res.locals.graphFootball= graph;
  next()
})
*/

router.get('/', function(req, res, next) {
  res.render('index', { title: 'Express', 
                        data: res.locals.graphFootball,
                        graphTools: graphTools
                      })
});
                    

module.exports = router;
