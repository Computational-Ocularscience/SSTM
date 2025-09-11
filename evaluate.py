import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'core'))
from PIL import Image
import argparse
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import imageio
import datasets
from utils import flow_viz
from utils import frame_utils
import csv
from glob import glob
import os.path as osp
import cv2
from core.utils import frame_utils

from sstm import SSTM
from utils.utils import InputPadder, forward_interpolate


@torch.no_grad()
def warp_image_with_flow(image, flow):
    """
    Warp an image using the provided optical flow field.

    :param image: Input image to be warped (H, W, 3) or (H, W)
    :param flow: Optical flow field of size (H, W, 2)
    :return: Warped image
    """
    # Get image dimensions

    image = image.squeeze(0)  # Remove the batch dimension, shape becomes (3, h, w)
    image = image.permute(1, 2, 0).cpu().numpy()

    h, w = image.shape[:2]

    image= np.clip(image, 0, 255).astype(np.uint8)

    # Create a grid of coordinates corresponding to each pixel
    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h),indexing='ij')

    # Add flow to the grid coordinates
    map_x = (grid_x + flow[:, :, 0]).astype(np.float32)
    map_y = (grid_y + flow[:, :, 1]).astype(np.float32)

    # Remap the image using the flow field
    warped_image = cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    return warped_image

@torch.no_grad()
def get_reconstruction_error(img1, img2, flow):
    rec_image = warp_image_with_flow(img2, flow)

    img1 = img1.squeeze(0)  # Remove the batch dimension, shape becomes (3, 416, 416)
    img1 = img1.permute(1, 2, 0).cpu().numpy()
    img1 = np.clip(img1, 0, 255).astype(np.uint8)

    rec_err = np.sqrt(np.sum((img1 - rec_image) ** 2))
    return rec_err

@torch.no_grad()
def create_sintel_submission(model, iters=32, warm_start=False, output_path='sintel_submission'):
    """ Create submission for the Sintel leaderboard """
    model.eval()
    for dstype in ['final', 'clean']:
        test_dataset = datasets.MpiSintel(split='test', aug_params=None, dstype=dstype)
        
        flow_prev, sequence_prev = None, None

        for test_id in range(len(test_dataset)):

            img1, img2, img3, (sequence, frame) = test_dataset[test_id]
            if sequence != sequence_prev:
                flow_prev = None
            
            padder = InputPadder(img1.shape)

            image1, image2, image3 = padder.pad(img1[None].cuda(), img2[None].cuda(), img3[None].cuda())

            start = time.time()

            flow_low1, flow_low2, flow_pr1, flow_pr2= model(image1, image2, image3, iters=iters, flow_init=flow_prev, test_mode=True)

            end = time.time()

            print("test inference time = ", (end - start)/2)

            flow1 = padder.unpad(flow_pr1[0]).permute(1, 2, 0).cpu().numpy()
            flow2 = padder.unpad(flow_pr2[0]).permute(1, 2, 0).cpu().numpy()

            if warm_start:
                flow_prev1 = forward_interpolate(flow_low1[0])[None].cuda()
                flow_prev2 = forward_interpolate(flow_low2[0])[None].cuda()
                flow_prev = flow_prev2
            
            output_dir = os.path.join(output_path, dstype, sequence)
            output_file1 = os.path.join(output_dir, 'frame%04d.flo' % (frame+1))
            output_file2 = os.path.join(output_dir, 'frame%04d.flo' % (frame+2))

            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            frame_utils.writeFlow(output_file1, flow1)
            frame_utils.writeFlow(output_file2, flow2)
            sequence_prev = sequence


@torch.no_grad()
def create_sintel_submission_vis(model, warm_start=False, output_path='sintel_submission'):
    """ Create submission for the Sintel leaderboard """
    model.eval()
    for dstype in ['clean', 'final']:
        test_dataset = datasets.MpiSintel(split='test', aug_params=None, dstype=dstype)

        flow_prev, sequence_prev = None, None
        prev = []
        for test_id in range(len(test_dataset)):

            image1, image2, image3, (sequence, frame) = test_dataset[test_id]
            if sequence != sequence_prev:
                flow_prev = None
                prev = []

            padder = InputPadder(image1.shape)
            image1, image2, image3 = padder.pad(image1[None].cuda(), image2[None].cuda(), image3[None].cuda())

            flow_low1, flow_low2, flow_pr1, flow_pr2 = model(image1, image2, image3, iters=32, flow_init=flow_prev, test_mode=True)
            flow1 = padder.unpad(flow_pr1[0]).permute(1, 2, 0).cpu().numpy()
            flow2 = padder.unpad(flow_pr2[0]).permute(1, 2, 0).cpu().numpy()

            # Visualizations
            flow_img1 = flow_viz.flow_to_image(flow1)
            flow_img2 = flow_viz.flow_to_image(flow2)
            image11 = Image.fromarray(flow_img1)
            image22 = Image.fromarray(flow_img2)

            if not os.path.exists(f'vis_test/SSTM_T++nws/{dstype}/'):
                os.makedirs(f'vis_test/SSTM_T++nws/{dstype}/flow')

            # image.save(f'vis_test/ours/{dstype}/flow/{test_id}.png')
            image11.save(f'vis_test/SSTM_T++nws/{dstype}/flow/{2*test_id}.png')
            image22.save(f'vis_test/SSTM_T++nws/{dstype}/flow/{2*test_id+1}.png')


            if warm_start:
                flow_prev1 = forward_interpolate(flow_low1[0])[None].cuda()
                prev.append(flow_prev1)
                flow_prev2 = forward_interpolate(flow_low2[0])[None].cuda()
                prev.append(flow_prev2)

                flow_prev = flow_prev2

            output_dir = os.path.join(output_path, dstype, sequence)

            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            #frame_utils.writeFlow(output_file1, flow1)
            #frame_utils.writeFlow(output_file2, flow2)
            sequence_prev = sequence

@torch.no_grad()
def create_kitti_submission(model, iters=24, output_path='kitti_submission'):
    """ Create submission for the Sintel leaderboard """
    model.eval()
    test_dataset = datasets.KITTI(split='testing', aug_params=None)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for test_id in range(len(test_dataset)):
        image1, image2, image3, (frame_id, ) = test_dataset[test_id]
        padder = InputPadder(image1.shape, mode='kitti')

        image1, image2, image3 = padder.pad(image1[None].cuda(), image2[None].cuda(), image3[None].cuda())

        #_, _, flow_pr, _ = model(image1, image2, image3, iters=iters, test_mode=True) #b/n 1st and 2nd frames
        _, _, _, flow_pr = model(image1, image2, image3, iters=iters, test_mode=True)  #b/n 2nd and 3rd frames

        flow = padder.unpad(flow_pr[0]).permute(1, 2, 0).cpu().numpy()

        output_filename = os.path.join(output_path, frame_id)
        frame_utils.writeFlowKITTI(output_filename, flow)


@torch.no_grad()
def create_kitti_submission_vis(model, output_path='kitti_sub'):
    """ Create submission for the Sintel leaderboard """
    model.eval()
    test_dataset = datasets.KITTI(split='testing', aug_params=None)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for test_id in range(len(test_dataset)):
        image1, image2, image3, (frame_id,) = test_dataset[test_id]
        padder = InputPadder(image1.shape, mode='kitti')
        image1, image2, image3 = padder.pad(image1[None].cuda(), image2[None].cuda(), image3[None].cuda())

        _, _, _, flow_pr = model(image1, image2, image3, iters=24, test_mode=True)

        flow = padder.unpad(flow_pr[0]).permute(1, 2, 0).cpu().numpy()

        output_filename = os.path.join(output_path, frame_id)
        frame_utils.writeFlowKITTI(output_filename, flow)

        # Visualizations
        flow_img = flow_viz.flow_to_image(flow)
        image = Image.fromarray(flow_img)
        if not os.path.exists(f'vis_kitti/SSTM++_T'):
            os.makedirs(f'vis_kitti/SSTM++_T/flow')

        image.save(f'vis_kitti/SSTM++_T/flow/{test_id}.png')


@torch.no_grad()
def validate_monkaa(model, iters=24):
    """ Perform evaluation on the Monkaa (validation) split """
    model.eval()
    results = {}

    for dstype in ['frames_cleanpass', 'frames_finalpass']:
        epe_list = []
        val_dataset = datasets.Monkaa(split='validation', dstype=dstype)
        for val_id in range(len(val_dataset)):
            image1, image2, image3, flow_gt1, flow_gt2, _, _ = val_dataset[val_id]
            image1 = image1[None].cuda()
            image2 = image2[None].cuda()
            image3 = image3[None].cuda()

            padder = InputPadder(image1.shape)
            image1, image2, image3 = padder.pad(image1, image2, image3)

            flow_low1, flow_low2, flow_pr1, flow_pr2 = model(image1, image2, image3, iters=iters, test_mode=True)

            flow1 = padder.unpad(flow_pr1[0]).cpu()
            flow2 = padder.unpad(flow_pr2[0]).cpu()

            epe = torch.sum((flow1[0].cpu() - flow_gt1) ** 2, dim=0).sqrt()
            epe_list.append(epe.view(-1).numpy())

            epe = torch.sum((flow2[0].cpu() - flow_gt2) ** 2, dim=0).sqrt()
            epe_list.append(epe.view(-1).numpy())


        epe_all = np.concatenate(epe_list)

        epe = np.mean(epe_all)
        px1 = np.mean(epe_all < 1)
        px3 = np.mean(epe_all < 3)
        px5 = np.mean(epe_all < 5)


        print("Validation (%s) Monkaa EPE: %f, 1px: %f, 3px: %f, 5px: %f" % (dstype, epe, px1, px3, px5))
        results[dstype] = np.mean(epe_list)
    return results

@torch.no_grad()
def validate_sintel(model, iters=32):
    """ Peform validation using the Sintel (train) split """
    model.eval()
    results = {}
    for dstype in ['clean', 'final']:
        val_dataset = datasets.MpiSintel(split='training', dstype=dstype)

        epe_list = []
        t = []
        for val_id in range(len(val_dataset)):
            start = time.time()

            image1, image2, image3, flow_gt1, flow_gt2, _, _ = val_dataset[val_id]
            image1 = image1[None].cuda()
            image2 = image2[None].cuda()
            image3 = image3[None].cuda()

            padder = InputPadder(image1.shape)
            image1, image2, image3 = padder.pad(image1, image2, image3)

            flow_low1, flow_low2, flow_pr1, flow_pr2 = model(image1, image2, image3, iters=iters, test_mode=True)


            flow1 = padder.unpad(flow_pr1[0]).cpu()
            flow2 = padder.unpad(flow_pr2[0]).cpu()

            end = time.time()
            #print('TIME = ', (end - start))
            t.append(end - start)

            epe1 = torch.sum((flow1 - flow_gt1)**2, dim=0).sqrt()
            epe_list.append(epe1.view(-1).numpy())
            epe2 = torch.sum((flow2 - flow_gt2) ** 2, dim=0).sqrt()
            epe_list.append(epe2.view(-1).numpy())

        print('inferenceTime = ', np.sum(t) / len(epe_list))
        epe_all = np.concatenate(epe_list)
        epe = np.mean(epe_all)
        px1 = np.mean(epe_all<1)
        px3 = np.mean(epe_all<3)
        px5 = np.mean(epe_all<5)

        print("Validation (%s) EPE: %f, 1px: %f, 3px: %f, 5px: %f" % (dstype, epe, px1, px3, px5))
        results[dstype] = np.mean(epe_list)

    return results

@torch.no_grad()
def validate_speckle(model, iters=32):
    """ Peform validation using the Sintel (train) split """
    model.eval()
    #for dstype in ['clean', 'final']:
    val_dataset = datasets.Speckle(split='training')
    epe_list = []
    for val_id in range(len(val_dataset)):

        image1, image2, image3, flow_gt1, flow_gt2, _, _ = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()
        image3 = image3[None].cuda()

        padder = InputPadder(image1.shape)
        image1, image2, image3 = padder.pad(image1, image2, image3)

        flow_low1, flow_low2, flow_pr1, flow_pr2 = model(image1, image2, image3, iters=iters, test_mode=True)


        flow1 = padder.unpad(flow_pr1[0]).cpu()
        flow2 = padder.unpad(flow_pr2[0]).cpu()

        epe1 = torch.sum((flow1 - flow_gt1)**2, dim=0).sqrt()
        epe_list.append(epe1.view(-1).numpy())
        epe2 = torch.sum((flow2 - flow_gt2) ** 2, dim=0).sqrt()
        epe_list.append(epe2.view(-1).numpy())


    epe_all = np.concatenate(epe_list)
    epe = np.mean(epe_all)

    print("Validation speckle EPE: %f" % epe)
    return {'speckle': epe}


@torch.no_grad()
def validate_kitti(model, iters=24):
    """ Peform validation using the KITTI-2015 (train) split """
    model.eval()
    val_dataset = datasets.KITTI(split='training')

    out_list, epe_list = [], []
    for val_id in range(len(val_dataset)):
        image1, image2, image3, flow_gt, valid_gt = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()
        image3 = image3[None].cuda()

        padder = InputPadder(image1.shape, mode='kitti')
        image1, image2, image3 = padder.pad(image1, image2, image3 )


        #flow_low, _, flow_pr, _ = model(image1, image2, image3, iters=iters, test_mode=True)
        _, flow_low, _, flow_pr = model(image1, image2, image3, iters=iters, test_mode=True)

        flow = padder.unpad(flow_pr[0]).cpu()

        epe = torch.sum((flow - flow_gt)**2, dim=0).sqrt()
        mag = torch.sum(flow_gt**2, dim=0).sqrt()

        epe = epe.view(-1)
        mag = mag.view(-1)
        val = valid_gt.view(-1) >= 0.5

        out = ((epe > 3.0) & ((epe/mag) > 0.05)).float()
        epe_list.append(epe[val].mean().item())
        out_list.append(out[val].cpu().numpy())

    epe_list = np.array(epe_list)
    out_list = np.concatenate(out_list)

    epe = np.mean(epe_list)
    f1 = 100 * np.mean(out_list)

    print("Validation KITTI: %f, %f" % (epe, f1))
    return {'kitti-epe': epe, 'kitti-f1': f1}

@torch.no_grad()
def validate_vkitti(model, iters=24):
    """ Peform validation using the virtual vkitti (validation) split """
    model.eval()
    val_dataset = datasets.VirtualKITTI2(split='validation')

    out_list, epe_list = [], []
    for val_id in range(len(val_dataset)):
        image1, image2, image3, flow_gt1, flow_gt2, valid_gt1, valid_gt2 = val_dataset[val_id]

        image1 = image1[None].cuda()
        image2 = image2[None].cuda()
        image3 = image3[None].cuda()

        padder = InputPadder(image1.shape, mode='kitti')
        image1, image2, image3 = padder.pad(image1, image2, image3 )

        low_low1, flow_low2, flow_pr1, flow_pr2 = model(image1, image2, image3, iters=iters, test_mode=True)

        flow1 = padder.unpad(flow_pr1[0]).cpu()
        flow2 = padder.unpad(flow_pr2[0]).cpu()

        epe1 = torch.sum((flow1 - flow_gt1)**2, dim=0).sqrt()
        epe2 = torch.sum((flow2 - flow_gt2) ** 2, dim=0).sqrt()

        mag1 = torch.sum(flow_gt1**2, dim=0).sqrt()
        mag2 = torch.sum(flow_gt2** 2, dim=0).sqrt()

        epe1 = epe1.view(-1)
        epe2 = epe2.view(-1)

        mag1 = mag1.view(-1)
        mag2 = mag2.view(-1)

        val1 = valid_gt1.view(-1) >= 0.5
        val2 = valid_gt2.view(-1) >= 0.5

        out1 = ((epe1 > 3.0) & ((epe1 / mag1) > 0.05)).float()
        out2 = ((epe2 > 3.0) & ((epe2 / mag2) > 0.05)).float()

        epe_list.append(epe1[val1].mean().item())
        epe_list.append(epe2[val2].mean().item())
        out_list.append(out1[val1].cpu().numpy())
        out_list.append(out2[val2].cpu().numpy())

    epe_list = np.array(epe_list)
    out_list = np.concatenate(out_list)

    epe = np.mean(epe_list)
    f1 = 100 * np.mean(out_list)

    print("Validation vKITTI: %f, %f" % (epe, f1))
    return {'kitti-epe': epe, 'kitti-f1': f1}

@torch.no_grad()
def validate_chairs(model, iters=24):
    """ Perform evaluation on the FlyingChairs (test) split """
    model.eval()
    epe_list = []

    val_dataset = datasets.FlyingChairs(split='validation')
    for val_id in range(len(val_dataset)):
        image1, image2, flow_gt, _ = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        _, flow_pr = model(image1, image2, iters=iters, test_mode=True)
        epe = torch.sum((flow_pr[0].cpu() - flow_gt)**2, dim=0).sqrt()
        epe_list.append(epe.view(-1).numpy())

    epe = np.mean(np.concatenate(epe_list))
    print("Validation Chairs EPE: %f" % epe)
    return {'chairs': epe}


@torch.no_grad()
def create_retinal_flow(model, iters=24, output_path='datasets/Retinal', dataset='DIGS'):
    """ optical flow estimates of ONH images"""
    model.eval()

    test_dataset = datasets.Retinal(split='test', aug_params=None, dataset=dataset)
    flow_prev, sequence_prev = None, None
    no_of_folders = len(test_dataset)

    with open(osp.join(output_path, 'full_rec_error_.csv'), mode='w', newline='') as file1:
        writer1 = csv.writer(file1)
        headers1 = ['Sequences', 'Rec_err1','Rec_err2']
        writer1.writerow(headers1)

        for test_id in range(no_of_folders):
            print(f"{test_id+1} out of {no_of_folders}")
            img1, img2, img3, sequence = test_dataset[test_id]

            if sequence != sequence_prev:
                flow_prev = None

            padder = InputPadder(img1.shape)
            image1, image2, image3 = padder.pad(img1[None].cuda(), img2[None].cuda(), img3[None].cuda())

            flow_low1, flow_low2, flow_pr1, flow_pr2 = model(image1, image2, image3, iters=iters, flow_init=flow_prev, test_mode=True)


            flow_GT = False
            if flow_GT:
                flow_gt1 = frame_utils.read_gen(osp.join(output_path, 'Flow', sequence, 'flow001.flo'))
                flow_gt2 = frame_utils.read_gen(osp.join(output_path, 'Flow', sequence, 'flow002.flo'))

            flow1 = padder.unpad(flow_pr1[0]).permute(1, 2, 0).cpu().numpy()
            flow2 = padder.unpad(flow_pr2[0]).permute(1, 2, 0).cpu().numpy()

            flow1_out = flow1
            flow2_out = flow2

            output_dir = os.path.join(output_path, dataset, sequence)
            output_file1 = os.path.join(output_dir, 'flow%04d.flo' % 1)
            output_file2 = os.path.join(output_dir, 'flow%04d.flo' % 2)

            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            frame_utils.writeFlow(output_file1, flow1_out)
            frame_utils.writeFlow(output_file2, flow2_out)
            sequence_prev = sequence


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--dataset', help="dataset for evaluation")
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')

    parser.add_argument('--num_heads', default=1, type=int,
                        help='number of heads in attention and aggregation')
    parser.add_argument('--position_only', default=False, action='store_true',
                        help='only use position-wise attention')
    parser.add_argument('--position_and_content', default=False, action='store_true',
                        help='use position and content-wise attention')

    args = parser.parse_args()

    #modified
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    #md
    model = torch.nn.DataParallel(SSTM(args))
    model.load_state_dict(torch.load(args.model, map_location=device))
    #model.load_state_dict(torch.load(args.model, map_location= torch.device('cpu')))

    model.cuda()
    model.eval()
    #Uncomment one of the following four lines depending on what you want to do on KITTI and Sintel test dataset
    #create_sintel_submission_vis(model.module, warm_start=False)
    #create_kitti_submission_vis(model.module)

    #create_sintel_submission(model.module, warm_start=False)
    #create_kitti_submission(model.module)
    if args.dataset == 'DIGS' or args.dataset == 'LEGS' or args.dataset == 'SPECKLE':
        create_retinal_flow(model.module, dataset=args.dataset)


    with torch.no_grad():
        if args.dataset == 'chairs':
            validate_chairs(model.module)
        elif args.dataset == 'monkaa':
            validate_monkaa(model.module)
        elif args.dataset == 'sintel':
            validate_sintel(model.module)
        elif args.dataset == 'speckle':
            validate_sintel(model.module)
        elif args.dataset == 'kitti':
            validate_kitti(model.module)
        elif args.dataset == 'vkitti':
            validate_vkitti(model.module)


