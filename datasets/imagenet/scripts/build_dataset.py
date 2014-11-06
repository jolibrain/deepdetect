import os, argparse

def convert_imageset(convert_imageset,images,prefix,vt='train',resize=256):
    cmd = convert_imageset + ' --resize_height=' + str(resize) + ' --resize_width=' + str(resize) + ' --shuffle ' + images + '/' + vt + '/ ' + images + '/' + vt + '.txt ' + prefix + '_' + vt + '_lmdb'
    print cmd
    os.system(cmd)

def compute_image_mean(image_mean,prefix,vt='train'):
    cmd = image_mean + ' ' + prefix + '_' + vt + '_lmdb ' + prefix + '_mean.binaryproto'
    print cmd
    os.system(cmd)

parser = argparse.ArgumentParser(description='Imagenet dataset builder')
parser.add_argument('--convert_imageset',type=str,default='build/tools/convert_imageset',help='location of the convert_imageset exe')
parser.add_argument('--image_mean',type=str,default='build/tools/compute_image_mean',help='location of the compute_image_mean exe')
parser.add_argument('--images',type=str,default='.',help='location of the dataset of images')
parser.add_argument('--prefix',type=str,default='img_dataset',help='dataset prefix name')
parser.add_argument('--repo',type=str,default='',help='locatation of the dataset of lmdb databases output repositories')
args = parser.parse_args()

convert_imageset(args.convert_imageset,args.images,args.prefix,'train',256)
convert_imageset(args.convert_imageset,args.images,args.prefix,'val',256)
compute_image_mean(args.image_mean,args.prefix,'train')
