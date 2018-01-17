#!/bin/bash
curl -X POST "http://localhost:$1/predict" -d '{
       "service":"imageserv",
       "parameters":{
         "output":{
           "rois":"'$2'"
      }
},
       "data":["'$3'"]
}'
