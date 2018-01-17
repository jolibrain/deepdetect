#!/bin/bash
curl -X PUT "http://localhost:$1/services/imageserv" -d '{
       "mllib":"caffe",
       "description":"object detection service",
       "type":"supervised",
       "parameters":{
          "input":{
            "connector":"image",
            "height": 300,
            "width": 300
          },
          "mllib":{
            "nclasses":21
          }
       },
       "model":{
          "repository":"'$2'"
        }
     }'
