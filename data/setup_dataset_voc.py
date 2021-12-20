import tarfile
import wget
import shutil
import os
import numpy as np
import scipy.io as sio
from shutil import copyfile
from glob import glob
from PIL import Image


def makedir(path):
  if not os.path.exists(path):
    os.makedirs(path)

def download_voc(path):
  url = 'http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar'
  fname = wget.download(url)
  with tarfile.open(fname) as tar:
    tar.extractall(path=path)
  if os.path.exists(fname):
    os.remove(fname)

def download_extra(path):
  url = 'http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/semantic_contours/benchmark.tgz'
  fname = wget.download(url)
  archive_name = os.path.join(path, fname)
  extra_folder_name = os.path.join(path, 'benchmark')
  makedir(extra_folder_name)
  with tarfile.open(fname) as tar:
    tar.extractall(path=os.path.join(path, 'benchmark'))
  if os.path.exists(fname):
    os.remove(fname)

def create_extra_folder():
  makedir(os.path.join('data/VOCdevkit/VOC2012/Extra'))
  copyfile(
      'data/palette.npy',
      'data/VOCdevkit/VOC2012/Extra/palette.npy'
  )
  copyfile(
      'data/train_extra_annot.txt',
      'data/VOCdevkit/VOC2012/Extra/train_extra_annot.txt'
  )

if __name__ == '__main__':

  DATASETS_ROOT = 'data'

  print('1. Downloading the VOC dataset')
  makedir(DATASETS_ROOT)
  download_voc(DATASETS_ROOT)
  create_extra_folder()

  # List of files that have extra annotations is placed in the dataset folder
  print('2. Locating the files')
  extra_annot_dir = os.path.join(DATASETS_ROOT, 'VOCdevkit/VOC2012/ImageSets/SegmentationAug/')
  makedir(extra_annot_dir)
  copyfile(os.path.join(DATASETS_ROOT, 'train_extra_annot.txt'),
          os.path.join(extra_annot_dir, 'train.txt'))
  
  # Downloading extra data and extracting it
  print('3. Downloading extra data')
  download_extra(DATASETS_ROOT)

  # Extracting extra annotations to the dataset folder
  print('4. Converting data to .png and saving to the dataset folder')
  extra_annot_folder = os.path.join(DATASETS_ROOT, 'VOCdevkit/VOC2012/SegmentationClassAug/')
  folder_name = os.path.join(os.path.join(DATASETS_ROOT, 'benchmark'), 'benchmark_RELEASE/dataset/cls')
  filenames = glob(os.path.join(folder_name, '*.mat'))
  makedir(extra_annot_folder)

  palette = np.load(os.path.join(DATASETS_ROOT, 'palette.npy')).tolist()

  print('5. Saving the extra dataset annotations')
  for i in range(len(filenames)):
      filename = filenames[i]
      name = filename.split('/')[-1].split('.')[0]
      mat = sio.loadmat(filename)['GTcls'][0][0][1]
      mask = Image.fromarray(mat)
      mask.putpalette(palette)
      mask.save(os.path.join(extra_annot_folder, name + '.png'), 'PNG')

  print('6. Removing extra folders and files')
  shutil.rmtree('data/benchmark')

  print('Pascal VOC 2012 dataset downloaded and setup. Process finished!')