import argparse
import cv2
import glob
import numpy as np
import os
import torch
from basicsr.utils import imwrite
from PIL import Image

from gfpgan import GFPGANer

conf = {'upscale': 2, # The final upsampling scale of the image
        'arch': 'clean', # The GFPGAN architecture. Option: clean | original
        'channel':2, # Channel multiplier for large networks of StyleGAN2
        'model_path': 'experiments/pretrained_models/GFPGANCleanv1-NoCE-C2.pth',
        'bg_upsampler': 'realesrgan', # background upsampler - CPU -> none
        'bg_tile': 400, # Tile size for background sampler, 0 for no tile during testing
        'test_path': 'inputs/whole_imgs', # Input folder
        'only_center_face': 'store_true', # Only restore the center face
        'aligned': 'store_true', # Input are aligned faces
        'paste_back': 'store_false', # Paste the restored faces back to images
        'save_root': 'results', # Path to save root
        }

def main():
    """Inference demo for GFPGAN.
    """

    img_path = 'inputs/cropped_faces/Adele_crop.png'
    # set up GFPGAN restorer
    restorer = GFPGANer(
        model_path=conf['model_path'],
        upscale=conf['upscale'],
        arch=conf['arch'],
        channel_multiplier=conf['channel'],
        bg_upsampler=None)

    # read image
    img_name = os.path.basename(img_path)
    print(f'Processing {img_name} ...')
    basename, ext = os.path.splitext(img_name)
    input_img = cv2.imread(img_path, cv2.IMREAD_COLOR)

    input_img = Image.open(img_path)
    input_img = cv2.cvtColor(np.array(input_img), cv2.COLOR_RGB2BGR)

    # restore faces and background if necessary
    cropped_faces, restored_faces, restored_img = restorer.enhance(
        input_img, has_aligned=conf['aligned'], only_center_face=conf['only_center_face'], paste_back=conf['paste_back'])

    for idx, (cropped_face, restored_face) in enumerate(zip(cropped_faces, restored_faces)):
        imwrite(restored_face, '/Users/nyt21/Devel/GFPGAN/GFPGAN/results/myresult.png')

    # # save faces
    # for idx, (cropped_face, restored_face) in enumerate(zip(cropped_faces, restored_faces)):
    #     # save cropped face
    #     save_crop_path = os.path.join(conf['save_root'], 'cropped_faces', f'{basename}_{idx:02d}.png')
    #     print('save_crop_path is {0}'.format(save_crop_path))
    #     imwrite(cropped_face, save_crop_path)
    #     # save restored face
    #     save_face_name = f'{basename}_{idx:02d}.png'
    #     save_restore_path = os.path.join(conf['save_root'], 'restored_faces', save_face_name)
    #     imwrite(restored_face, save_restore_path)
    #     # save comparison image
    #     cmp_img = np.concatenate((cropped_face, restored_face), axis=1)
    #     imwrite(cmp_img, os.path.join(conf['save_root'], 'cmp', f'{basename}_{idx:02d}.png'))

    # save restored img
    # if restored_img is not None:
    #     extension = ext[1:]

    #     save_restore_path = os.path.join(args.save_root, 'restored_imgs', f'{basename}.{extension}')
    #     imwrite(restored_img, save_restore_path)

if __name__ == '__main__':
    main()
