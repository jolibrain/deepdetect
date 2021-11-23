# Predict video connector (inherits image connector)

Compatible with models trained with image connector

## /resources

`PUT /resources/{resource_id}`

*Parameters (video):*

- `type`:
    - video
- `source`:
    - youtube video (needs youtube-dl?)
    - stream url
    - gstreamer pipeline
    - video file
    - local camera
- `backend`: Use specific backend to read the video
    - auto (default)
    - ffmpeg
    - gstreamer
    - v4l2
<!-- (- `realtime`: if true, allow to skip frames if processing is not as fast as original material) -->

*Example:*
```bash
# resource creation creation
curl -X PUT "http://localhost:8080/resources/yt_video" -d '{
    "type": "video",
    "backend": "auto", 
    "source": "https://www.youtube.com/watch?v=XnZH4izf_rI"
}
```

*Response:*
```json
{
   "status":{
      "code":201,
      "msg":"OK"
   },
   "head":{
      "method":"/resources",
      "id": "yt_video",
      "time":74.0
   }
}
```

- `id`: id of created resource

## /stream

`PUT /stream/{stream_id}`

*Parameters:*
- `predict`: Encapsulated predict call (as in `/predict`)
- `chain`: Encapsulated chain call (as in `/chain`)
- `output`: Parameters for streaming out
- `output.video_out`:
    - gstreamer pipeline
    - video file
    - other streams... ?
- `output.type`: type of sink
    - video
    - gst_pipeline -> subtype of video?
- `output.backend`: video backend (opencv) used to encode the video
    

*Example:*
```bash
curl -X PUT "http://localhost:8080/stream/my_realtime" -d '{
   "predict": {
       "service":"ganserv",
       "parameters":{
           "mllib": {
               "extract_layer": "last"
           }
       },
       "data":["file_video"]
   }
   "output": {
       "type": "video",
       "backend": "gstreamer",
       "video_out":  "/opt/platform/data/video_out.mp4"
   }
}'
```

<!--
gstreamer pipeline: "appsrc ! queue ! videoconvert ! video/x-raw,format=I420,width=1920,height=1080, framerate=30/1 ! jpegenc ! rtpjpegpay ! queue ! udpsink host=localhost port=5000"
-->

*Response:*
```json
{
   "status":{
      "code":201,
      "msg":"Created"
   },
   "head":{
      "method":"/stream",
      "id":"my_realtime",
      "status":"running"
   }
}
```

`GET /stream/{stream_id}`

*Example:*

```bash
curl -X GET "http://localhost:8080/stream/my_realtime"
```

*Response:*

```json
{
   "status":{
      "code":200,
      "msg":"OK"
   },
   "head":{
      "method":"/predict",
      "id":"my_realtime",
      "status":"running",
      "time":74.0
   },
   "body":{
     "processed_frames": 2000,
     "elapsed_ms": 74999,
     "fps": 12.0,
     "remaining_time_ms": 89333,
     "video": {
        "width": 1920,
        "height": 1080,
        "fps": 30
     }
   }
}
```

If stream is stopped:
```json
{
   "status":{
      "code":200,
      "msg":"OK"
   },
   "head":{
      "method":"/predict",
      "job":"my_realtime",
      "status":"terminated",
      "time":567.0
   },
   "body":{
     "processed_frames": 23456,
     "elapsed_ms": 45678,
     "fps": 12.0,
     "remaining_time_ms": 89333,
     "video": {
        "width": 1920,
        "height": 1080,
        "fps": 30
     }
   }
}
```

- `head.status`: one of
    - running
    - terminated
    - error

When the stream is consumed entirely, the resource is deleted.

## Video to data stream (sync)

```bash
# resource creation
curl -X PUT "http://localhost:8080/resources/yt_video" -d '{
    "type": "video",
    "source": "https://www.youtube.com/watch?v=XnZH4izf_rI"
}

# service creation
curl -X PUT "http://localhost:8080/services/imageserv" -d '{
    "description": "image detection service",
    "mllib": "caffe",
    "model": {
        "init": "https://deepdetect.com/models/init/desktop/images/detection/detection_voc0712.tar.gz",
        "repository": "/opt/models/detection_voc0712",
        "create_repository": true
    },
    "parameters": {
        "input": {
            "connector": "video"
        }
    },
    "type": "supervised"
}
'

# predict call
curl -X POST "http://localhost:8080/predict" -d '{
       "service":"imageserv",
       "parameters":{
         "output":{
           "bbox": true,
           "confidence_threshold": 0.1
         }
       },
       "data":["resource:yt_video"]
     }'
```

*Response:*
```json
{
   "status":{
      "code":200,
      "msg":"OK"
   },
   "head":{
      "method":"/predict",
      "time":1591.0,
      "service":"imageserv"
   },
   "body":{
      "predictions":{
         "uri":"resource:yt_video/34",
         "classes":[
           {
              "bbox": {
                  "ymax": 2101.48583984375,
                  "xmax": 3836.814697265625,
                  "ymin": 1399.1077880859375,
                  "xmin": 2862.972412109375
               },
               "prob": 0.9999996423721313,
               "cat": "1"
           } 
         ]
      }
   }
}
```
- `body.predictions.uri`: appends frame id to resource id

*Last frame response:*
```json
{
   "status":{
      "code":200,
      "msg":"OK"
   },
   "head":{
      "method":"/predict",
      "time":1591.0,
      "service":"imageserv",
      "last": true
   },
   "body":{
      "predictions":{
         "uri":"resource:yt_video/3444",
         "classes":[
           {
              "bbox": {
                  "ymax": 2101.48583984375,
                  "xmax": 3836.814697265625,
                  "ymin": 1399.1077880859375,
                  "xmin": 2862.972412109375
               },
               "prob": 0.9999996423721313,
               "cat": "1"
           } 
         ]
      }
   }
}
```

*No more frames:*
```json
{"status":{"code":404,"msg":"No more frames in resources"}}
```

```bash
# close video
curl -X DELETE "http://localhost:8080/resources/yt_video"
```

```json
{"status":{"code":200,"msg":"OK"}}
```

*Notes:*
- 1 datastream = 1 service
- Create service with video connector (it could be created with img_connector)
- *future* use websockets to stream results?

## Video to Video (GAN, async) 

```bash
# create input video
curl -X PUT "http://localhost:8080/resources/file_video" -d '{
    "type": "video",
    "source": "/opt/platform/data/video.mp4"
}
```

*Response:*
```json
{
   "status":{
      "code":201,
      "msg":"OK"
   },
   "head":{
      "method":"/resources",
      "id": "file_video",
      "time":74.0
   }
}
```

```bash
# service creation
curl -X PUT "http://localhost:8080/services/ganserv" -d '{
    "description": "image classification service",
    "mllib": "torch",
    "model": {
        "repository": "/opt/models/cat_dogs_gan"
    },
    "parameters": {
        "input": {
            "connector": "image"
        },
        "mllib": {
            "extract_layer": "last"
        }
    },
    "type": "supervised"
}
'
```

*Notes:*
- Similar to a simple image gan service

```bash
# first predict
curl -X PUT "http://localhost:8080/stream/my_realtime" -d '{
   "predict": {
       "service":"ganserv",
       "parameters":{
           "mllib": {
               "extract_layer": "last"
           }
       },
       "data":["file_video"]
   }
   "output": {
       "type": "video",
       "backend": "gstreamer",
       "video_out": "/opt/platform/data/video_out.mp4"
    }
}'
```

*Response:*
```json
{
   "status":{
      "code":201,
      "msg":"Created"
   },
   "head":{
      "method":"/stream",
      "id":"my_realtime",
      "status":"running"
   }
}
```

```bash
# get information on predict call
curl -X GET "http://localhost:8080/stream/my_realtime"

# Video is stopped when stream stops, or when deleting the service
```

```json
{
   "status":{
      "code":200,
      "msg":"OK"
   },
   "head":{
      "method":"/stream",
      "job":"my_realtime",
      "status":"running",
      "time":74.0
   },
   "body":{
     "processed_frames": 2000,
     "elapsed_ms": 74999,
     "fps": 12.0,
     "remaining_time_ms": 89333,
     "video": {
        "width": 1920,
        "height": 1080,
        "fps": 30
     }
   }
}
```
