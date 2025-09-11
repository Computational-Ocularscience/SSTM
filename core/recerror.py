import os
import numpy as np
import torch
import datasets
import csv
import os.path as osp
import cv2
from utils import frame_utils


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
    #print('img1: ', np.mean(image))

    image= np.clip(image, 0, 255).astype(np.uint8)

    # Create a grid of coordinates corresponding to each pixel
    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))

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
def create_retinal_flow(model, iters=24, output_path='datasets/Retinal', dataset='example'):
    """ optical flow estimates of ONH images"""
    model.eval()

    test_dataset = datasets.Retinal(split='test', aug_params=None, dataset=dataset)
    flow_prev, sequence_prev = None, None
    no_of_folders = len(test_dataset)
    for test_id in range(no_of_folders):
        print(f"{test_id} out of {no_of_folders}")
        img1, img2, img3, sequence = test_dataset[test_id]

        if sequence != sequence_prev:
            flow_prev = None

        padder = InputPadder(img1.shape)
        image1, image2, image3 = padder.pad(img1[None].cuda(), img2[None].cuda(), img3[None].cuda())

        flow_low1, flow_low2, flow_pr1, flow_pr2, flow_predictions1, flow_predictions2 = model(image1, image2, image3, iters=iters, flow_init=flow_prev,
                                                             test_mode=True)
        min_err1 = float('inf')
        min_err2 = float('inf')

        flow_GT = False

        if flow_GT:
            flow_gt1 = frame_utils.read_gen(osp.join(output_path, 'Flow', sequence, 'flow001.flo'))
            flow_gt2 = frame_utils.read_gen(osp.join(output_path, 'Flow', sequence, 'flow002.flo'))


        with open( osp.join(output_path, dataset, sequence, 'rec_error_output.csv'), mode='w', newline='') as file:
            writer = csv.writer(file)
            if flow_GT:
                headers = ['Iteration', 'EPE1', 'Rec_err1', 'EPE2', 'Rec_err2']
            else:
                headers = ['Iteration', 'Rec_err1', 'Rec_err2']
            writer.writerow(headers)

            for i in range(iters):
                flow_pr1 = flow_predictions1[i]
                flow_pr2 = flow_predictions2[i]

                flow1 = padder.unpad(flow_pr1[0]).permute(1, 2, 0).cpu().numpy()
                flow2 = padder.unpad(flow_pr2[0]).permute(1, 2, 0).cpu().numpy()

                rec_err1 = get_reconstruction_error(img1, img2, flow1)
                rec_err2 = get_reconstruction_error(img2, img3, flow2)

                if rec_err1 < min_err1:
                    flow1_out = flow1
                    min_err1 = rec_err1
                if rec_err2 < min_err2:
                    flow2_out = flow2
                    min_err2 = rec_err2
                if flow_GT:
                    epe1 = np.sqrt(np.sum((flow1 - flow_gt1) ** 2, axis=2))
                    epe1 = np.mean(epe1)

                    epe2 = np.sqrt(np.sum((flow2 - flow_gt2) ** 2, axis=2))
                    epe2 = np.mean(epe2)

                    row = [i+1, epe1, rec_err1, epe2, rec_err2]
                else:
                    row = [i+1, rec_err1, rec_err2]

                writer.writerow(row)

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