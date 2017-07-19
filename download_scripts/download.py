#!/usr/bin/env python
# coding=utf-8


"""
Modification of https://github.com/stanfordnlp/treelstm/blob/master/scripts/download.py
Downloads the following:
- point cloud data version of modelnet40 dataset
"""

from __future__ import print_function
import os
import sys
import gzip
import json
import shutil
import zipfile
import argparse
import subprocess
from six.moves import urllib

from IPython.core.debugger import Tracer 
debug_here = Tracer() 

parser = argparse.ArgumentParser(description='Download dataset for pointnet.')
parser.add_argument('--dataset', required=True, help=' can be [modelnet40_pcl, shapenetcore_partanno, ??]')

def download(url, dirpath):
    filename = url.split('/')[-1]
    filepath = os.path.join(dirpath, filename)
    # print(url)
    u = urllib.request.urlopen(url)
    # './datasets/modelnet40_ply_hdf5_2048.zip'
    # print(filepath)
    f = open(filepath, 'wb')
    filesize = int(u.headers["Content-Length"])
    print("Downloading: %s Bytes: %s" % (filename, filesize))

    downloaded = 0
    block_sz = 8192
    status_width = 70
    while True:
        buf = u.read(block_sz)
        if not buf:
            print('')
            break
        else:
            print('', end='\r')
        downloaded += len(buf)
        f.write(buf)
        status = (("[%-" + str(status_width + 1) + "s] %3.2f%%") %
            ('=' * int(float(downloaded) / filesize * status_width) + '>', downloaded * 100. / filesize))
        print(status, end='')
        sys.stdout.flush()
    f.close()
    return filepath

def unzip(filepath):

    print("Extracting: " + filepath)
    dirpath = os.path.dirname(filepath)
    with zipfile.ZipFile(filepath) as zf:
        zf.extractall(dirpath)
    os.remove(filepath)


def download_modelnet40_pcl(dirpath): 
    """
    dirpath = './datasets/'
    """ 
    data_folder = 'modelnet40_pcl'
    data_dir = os.path.join(dirpath, data_folder)
    if os.path.exists(data_dir):
        print('Found modelnet40_pcl - skip')
        return 
    url = 'https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip' 
    file_path = download(url, dirpath)
    unzip(file_path)

# download original shapenetcore_partanno dataset 
def download_shapenetcore_partanno(dirpath): 
    """
    dirpath = './datasets/'
    """ 
    data_folder = 'shapenetcore_partanno'
    data_dir = os.path.join(dirpath, data_folder)
    if os.path.exists(data_dir):
        print('Found shapenetcore_partanno - skip')
        return 
    url = 'https://shapenet.cs.stanford.edu/ericyi/shapenetcore_partanno_v0.zip' 
    file_path = download(url, dirpath)
    unzip(file_path)

# download original shapenetcore_partanno dataset 
def download_shapenetcore_partanno_h5(dirpath): 
    """
    dirpath = './datasets/'
    """ 
    data_folder = 'shapenet_part_seg_hdf5_dataset'
    data_dir = os.path.join(dirpath, data_folder)
    '''
    if os.path.exists(data_dir):
        print('Found shapenetcore_partanno_h5 - skip')
        return
    ''' 
    url = 'https://shapenet.cs.stanford.edu/media/shapenet_part_seg_hdf5_data.zip' 
    # file_path = download(url, dirpath)
    filename = url.split('/')[-1]
    file_path = os.path.join(dirpath, filename)
    # print(url)
    unzip(file_path)

# download original shapenetcore_partanno dataset 
def download_shapenet_partanno_seg_bench_v0(dirpath): 
    """
    dirpath = './datasets/'
    """ 
    data_folder = 'shapenetcore_partanno_segmentation_benchmark_v0'
    data_dir = os.path.join(dirpath, data_folder)
    if os.path.exists(data_dir):
        print('Found shapenetcore_partanno_h5 - skip')
        return
    url = 'https://shapenet.cs.stanford.edu/ericyi/shapenetcore_partanno_segmentation_benchmark_v0.zip' 
    file_path = download(url, dirpath)
    # print(url)
    unzip(file_path)




def download_celeb_a(dirpath):
    data_dir = 'celebA'
    if os.path.exists(os.path.join(dirpath, data_dir)):
        print('Found Celeb-A - skip')
        return
    url = 'https://www.dropbox.com/sh/8oqt9vytwxb3s4r/AADIKlz8PR9zr6Y20qbkunrba/Img/img_align_celeba.zip?dl=1&pv=1'
    filepath = download(url, dirpath)
    zip_dir = ''
    with zipfile.ZipFile(filepath) as zf:
        zip_dir = zf.namelist()[0]
        zf.extractall(dirpath)
    os.remove(filepath)
    os.rename(os.path.join(dirpath, zip_dir), os.path.join(dirpath, data_dir))

def _list_categories(tag):
    url = 'http://lsun.cs.princeton.edu/htbin/list.cgi?tag=' + tag
    f = urllib.request.urlopen(url)
    return json.loads(f.read())

def _download_lsun(out_dir, category, set_name, tag):
    url = 'http://lsun.cs.princeton.edu/htbin/download.cgi?tag={tag}' \
          '&category={category}&set={set_name}'.format(**locals())
    print(url)
    if set_name == 'test':
        out_name = 'test_lmdb.zip'
    else:
        out_name = '{category}_{set_name}_lmdb.zip'.format(**locals())
    out_path = os.path.join(out_dir, out_name)
    cmd = ['curl', url, '-o', out_path]
    print('Downloading', category, set_name, 'set')
    subprocess.call(cmd)

def download_lsun(dirpath):
    data_dir = os.path.join(dirpath, 'lsun')
    if os.path.exists(data_dir):
        print('Found LSUN - skip')
        return
    else:
        os.mkdir(data_dir)

    tag = 'latest'
    #categories = _list_categories(tag)
    categories = ['bedroom']

    for category in categories:
        _download_lsun(data_dir, category, 'train', tag)
        _download_lsun(data_dir, category, 'val', tag)
    _download_lsun(data_dir, '', 'test', tag)

def download_mnist(dirpath):
    data_dir = os.path.join(dirpath, 'mnist')
    if os.path.exists(data_dir):
        print('Found MNIST - skip')
        return
    else:
        os.mkdir(data_dir)
    url_base = 'http://yann.lecun.com/exdb/mnist/'
    file_names = ['train-images-idx3-ubyte.gz','train-labels-idx1-ubyte.gz','t10k-images-idx3-ubyte.gz','t10k-labels-idx1-ubyte.gz']
    for file_name in file_names:
        url = (url_base+file_name).format(**locals())
        print(url)
        out_path = os.path.join(data_dir,file_name)
        cmd = ['curl', url, '-o', out_path]
        print('Downloading ', file_name)
        subprocess.call(cmd)
        cmd = ['gzip', '-d', out_path]
        print('Decompressing ', file_name)
        subprocess.call(cmd)

def prepare_data_dir(path = './datasets/'):
    if not os.path.exists(path):
        os.mkdir(path)

if __name__ == '__main__':
    opt = parser.parse_args()
    prepare_data_dir()

    if opt.dataset == 'celebA':
        download_celeb_a('./datasets/')
    elif opt.dataset == 'lsun':
        download_lsun('./datasets/')
    elif opt.dataset == 'mnist':
        download_mnist('./datasets/')
    # for experiment on pointnet 
    elif opt.dataset == 'modelnet40_pcl':
        download_modelnet40_pcl('./datasets/')
    elif opt.dataset == 'shapenetcore_partanno':
        download_shapenetcore_partanno('./datasets/')
    elif opt.dataset == 'shapenetcore_partanno_h5':
        debug_here() 
        download_shapenetcore_partanno_h5('./datasets/')
    elif opt.dataset == 'shapenetcore_partanno_ben_v0':
        download_shapenet_partanno_seg_bench_v0('./datasets/')
    else: 
        print('not supported dataset dowloading') 


