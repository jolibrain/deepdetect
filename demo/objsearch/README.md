### Objsearch demo

copy or link VGG_VOC0712_SSD_300x300_iter_60000.caffemodel in ./model

launch sueprvised dede (can use launch_dede_detector.sh 8080 ./model)

to get roi_pool : launch roi request : use inspect_rois.sh 8080 rois ./velo2.jpg
"rois" is the name of the splitted roi info in deploy.prototxt



Notes : in 2017_12_14 deploy, roi_pool layer is fed only with conv4_3 (not relued!).

however, other outputs are possible, just need to aggregate somehow fc7 conv6_2 conv7_2 conv8_2 before feeding roi_pool

