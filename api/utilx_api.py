import argparse
import glob
import numpy as np
import os
import torch
from basicsr.utils import imwrite

from gfpgan import GFPGANer

def prepare_data():
    conf = {'upscale': 2, # The final upsampling scale of the image
            'arch': 'clean', # The GFPGAN architecture. Option: clean | original
            'channel':2, # Channel multiplier for large networks of StyleGAN2
            'model_path': '/Users/nyt21/Devel/GFPGAN/GFPGAN/experiments/pretrained_models/GFPGANCleanv1-NoCE-C2.pth',
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