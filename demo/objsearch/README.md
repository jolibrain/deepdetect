### Objsearch demo


Notes : in 2017_12_14 deploy, roi_pool layer is fed only with conv4_3 (not relued!).


very short use guide to object search:
copy or link VGG_VOC0712_SSD_300x300_iter_60000.caffemodel in ./model
launch dede
python objsearch.py --index your_images_repo
python objsearch.py --search your_image (velo2.jpg is given as an example)

for lower-level access (no indexing), you can use
copy or link VGG_VOC0712_SSD_300x300_iter_60000.caffemodel in ./model
launch dede
launch_dede_detector.sh 8080 absolute_path_to_model_dir
inspect_rois.sh 8080 rois absolute_path_to_image
