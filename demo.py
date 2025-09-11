import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'core'))

import argparse
import os
import cv2
import glob
import numpy as np
import torch
from PIL import Image

from sstm import SSTM
from utils import flow_viz
from utils.utils import InputPadder
import struct


DEVICE = 'cuda'

def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)

    if len(img.shape) == 2 or (len(img.shape) == 3 and img.shape[2] == 1):
        # Duplicate the channel to create a 3-channel image
        img = np.stack((img.squeeze(),) * 3, axis=-1)

    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)

def write_flow(flow, filename):
    """
    Write optical flow to a .flo file.

    Args:
    - flow: A numpy array of shape (H, W, 2) containing the flow vectors.
    - filename: The file path where the .flo file should be saved.
    """
    H, W, _ = flow.shape
    with open(filename, 'wb') as f:
        # Write the .flo file header
        f.write(b'PIEH')  # 4-byte header 'PIEH'
        f.write(struct.pack('i', W))
        f.write(struct.pack('i', H))

        # Write the flow data
        flow = flow.astype(np.float32)
        f.write(flow.tobytes())

def viz(img, flo):
    img = img[0].permute(1,2,0).cpu().numpy()
    flo = flo[0].permute(1,2,0).cpu().numpy()
    
    # map flow to rgb image
    flo = flow_viz.flow_to_image(flo)
    #img_flo = np.concatenate([img, flo], axis=0)
    img_flow = cv2.cvtColor(flo, cv2.COLOR_BGR2RGB)
    cv2.imwrite('flow__.jpg', img_flow)



def demo(args):
    model = torch.nn.DataParallel(SSTM(args))
    model.load_state_dict(torch.load(args.model))

    model = model.module
    model.to(DEVICE)
    model.eval()

    with torch.no_grad():
        images = glob.glob(os.path.join(args.path, '*.png')) + \
                 glob.glob(os.path.join(args.path, '*.jpg')) + \
                 glob.glob(os.path.join(args.path, '*.PNG'))

        
        images = sorted(images)


        for imfile1, imfile2, imfile3 in zip(images[0::2], images[1::2], images[2::2]):

            image1 = load_image(imfile1)
            image2 = load_image(imfile2)
            image3 = load_image(imfile3)


            padder = InputPadder(image1.shape)
            image1, image2, image3 = padder.pad(image1, image2, image3)

            flow_low1,flow_low2, flow_up1, flow_up2 = model(image1, image2, image3, iters=20, test_mode=True)
            #viz(image1, flow_up1)
            print(image1.shape)
            write_flow(flow_up1[0].permute(1, 2, 0).cpu().numpy(), 'flow0001.flo')
            #write_flow(flow_up2[0].permute(1, 2, 0).cpu().numpy(), 'flow0002.flo')
            viz(image2, flow_up1)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--path', help="dataset for evaluation")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')


    parser.add_argument('--num_heads', default=1, type=int,
                            help='number of heads in attention and aggregation')
    parser.add_argument('--position_only', default=False, action='store_true',
                            help='only use position-wise attention')
    parser.add_argument('--position_and_content', default=False, action='store_true',
                            help='use position and content-wise attention')

    args = parser.parse_args()
    demo(args)


