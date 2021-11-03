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
(- `realtime`: if true, allow to skip frames if processing is not as fast as original material)

## Video to data stream (sync)


*Notes:*
- 1 datastream = 1 service
- Create service with video connector (it could be created with img_connector)
- *future* use websockets to stream results?

```bash
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

# datastream creation
curl -X PUT "http://localhost:8080/resources/yt_video" -d '{
    "type": "video",
    "source": "https://www.youtube.com/watch?v=XnZH4izf_rI",
    "realtime": false
}


# first predict
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

# close video
curl -X DELETE "http://localhost:8080/resources/yt_video"
```

## Video to Video (GAN, async) 

*Parameters:*
- `video_out`:
    - gstreamer pipeline
    - video file
    - other streams... ?

*Notes:*
- Add async predict call and query predict status

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

# create input video
curl -X PUT "http://localhost:8080/resources/file_video" -d '{
    "type": "video",
    "source": "/opt/platform/data/video.mp4",
    "realtime": false
}

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
       "type": "gst_pipeline",
       "backend": "gstreamer",
       "video_out": "appsrc ! queue ! videoconvert ! video/x-raw,format=I420,width=1920,height=1080, framerate=30/1 ! jpegenc ! rtpjpegpay ! queue ! udpsink host=localhost port=5000"
    }
}'

# "video_out": /opt/platform/data/video_out.mp4

# get information on predict call
curl -X GET "http://localhost:8080/stream/my_realtime"

# Video is stopped when stream stops, or when deleting the service
```

`GET /stream` response:

```json
{
   "status":{
      "code":200,
      "msg":"OK"
   },
   "head":{
      "method":"/predict",
      "job":1,
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

