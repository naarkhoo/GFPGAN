import argparse
import glob
import numpy as np
import os
import sys
import requests
import torch
from basicsr.utils import imwrite

main_dir_path = "/".join(os.getcwd().split("/")[:-1])
sys.path.insert(1, main_dir_path)

from gfpgan import GFPGANer

def download_weights():
    weights_files = ['https://github.com/TencentARC/GFPGAN/releases/download/v0.2.0/GFPGANCleanv1-NoCE-C2.pth',
                     'https://github.com/TencentARC/GFPGAN/releases/download/v0.1.0/GFPGANv1.pth',
                     'https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth']
    for url in weights_files:
        filename = url.split('/')[-1]
        if not os.path.exists(main_dir_path + '/experiments/pretrained_models/' + filename):
            response = requests.get(url)
            if response.status_code == 200:
                print('started downloading {filename}'.format(filename = filename))
                with open(main_dir_path + '/experiments/pretrained_models/' + filename, 'wb') as f:
                    f.write(response.content)
            else:
                print('weight file {url} does not exist'.format(url = url))
        else:
            print('file {file} exist !'.format(file = main_dir_path + '/experiments/pretrained_models/' + filename))

def prepare_data():

    conf = {'upscale': 2, # The final upsampling scale of the image
            'arch': 'clean', # The GFPGAN architecture. Option: clean | original
            'channel':2, # Channel multiplier for large networks of StyleGAN2
            'model_path': main_dir_path + '/experiments/pretrained_models/GFPGANCleanv1-NoCE-C2.pth',
            'bg_upsampler': 'realesrgan', # background upsampler - CPU -> none
            'bg_tile': 400, # Tile size for background sampler, 0 for no tile during testing
            'test_path': 'inputs/whole_imgs', # Input folder
            'only_center_face': 'store_true', # Only restore the center face
            'aligned': 'store_true', # Input are aligned faces
            'paste_back': 'store_false', # Paste the restored faces back to images
            'save_root': 'results', # Path to save root
            }

    restorer = GFPGANer(
        model_path=conf['model_path'],
        upscale=conf['upscale'],
        arch=conf['arch'],
        channel_multiplier=conf['channel'],
        bg_upsampler=None)

    return(conf, restorer)
