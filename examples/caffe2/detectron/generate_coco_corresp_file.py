#!/usr/bin/env python

import argparse
import detectron.datasets.dummy_datasets as dummy_datasets

parser = argparse.ArgumentParser()
parser.add_argument('repo')
args = parser.parse_args()
classes = dummy_datasets.get_coco_dataset().classes
corresp = '\n'.join('{} {}'.format(i, classes[i]) for i, _ in enumerate(classes))
open(args.repo + '/corresp.txt', 'w').write(corresp)
