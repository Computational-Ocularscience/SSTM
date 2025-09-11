# Data loading based on https://github.com/NVIDIA/flownet2-pytorch

import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F

import os
import sys
import math
import random
from glob import glob
import os.path as osp

sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))
from utils import frame_utils
from utils.augmentor import FlowAugmentor, SparseFlowAugmentor


class FlowDataset(data.Dataset):
    def __init__(self, aug_params=None, sparse=False, flowgt='two'):
        self.augmentor = None
        self.sparse = sparse
        self.flowgt = flowgt
        #self.type = type
        if aug_params is not None:
            if sparse:
                self.augmentor = SparseFlowAugmentor(**aug_params)
            else:
                self.augmentor = FlowAugmentor(**aug_params)

        self.is_test = False
        self.init_seed = False
        self.flow_list = []
        self.image_list = []
        self.extra_info = []
        self.occ_list = None


    def __getitem__(self, index):

        if self.is_test:
            img1 = frame_utils.read_gen(self.image_list[index][0])
            img2 = frame_utils.read_gen(self.image_list[index][1])
            img3 = frame_utils.read_gen(self.image_list[index][2])

            #if self.type == 'gray':
            img1 = np.array(img1).astype(np.uint8)
            img2 = np.array(img2).astype(np.uint8)
            img3 = np.array(img3).astype(np.uint8)

            #if gray
            if len(img1.shape) == 2 or (len(img1.shape) == 3 and img1.shape[2] == 1):
                img1 = np.stack([img1, img1, img1], axis=-1)
                img2 = np.stack([img2, img2, img2], axis=-1)
                img3 = np.stack([img3, img3, img3], axis=-1)

            img1 = np.array(img1).astype(np.uint8)[..., :3]
            img2 = np.array(img2).astype(np.uint8)[..., :3]
            img3 = np.array(img3).astype(np.uint8)[..., :3]

            img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
            img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
            img3 = torch.from_numpy(img3).permute(2, 0, 1).float()


            return img1, img2, img3, self.extra_info[index]

        if not self.init_seed:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                torch.manual_seed(worker_info.id)
                np.random.seed(worker_info.id)
                random.seed(worker_info.id)
                self.init_seed = True

        index = index % len(self.image_list)


        valid = None
        if self.sparse and self.flowgt=='two':
            flow1, valid1 = frame_utils.readFlowVirtualKITTI (self.flow_list[index][0])
            flow2, valid2 = frame_utils.readFlowVirtualKITTI (self.flow_list[index][1])

            flow1 = np.array(flow1).astype(np.float32)
            flow2 = np.array(flow2).astype(np.float32)
        elif self.sparse and self.flowgt=='one':
            flow, valid = frame_utils.readFlowKITTI(self.flow_list[index][0])
            flow = np.array(flow).astype(np.float32)
        else:
            flow1 = frame_utils.read_gen(self.flow_list[index][0])
            flow2 = frame_utils.read_gen(self.flow_list[index][1])

            flow1 = np.array(flow1).astype(np.float32)
            flow2 = np.array(flow2).astype(np.float32)
        if self.occ_list is not None:
            occ1 = frame_utils.read_gen(self.occ_list[index][0])
            occ2 = frame_utils.read_gen(self.occ_list[index][1])

            occ1 = np.array(occ1).astype(np.uint8)
            occ2 = np.array(occ2).astype(np.uint8)

            occ1 = torch.from_numpy(occ1 // 255).bool()
            occ2 = torch.from_numpy(occ2 // 255).bool()

        img1 = frame_utils.read_gen(self.image_list[index][0])
        img2 = frame_utils.read_gen(self.image_list[index][1])
        img3 = frame_utils.read_gen(self.image_list[index][2])


        img1 = np.array(img1).astype(np.uint8)
        img2 = np.array(img2).astype(np.uint8)
        img3 = np.array(img3).astype(np.uint8)

        # grayscale images
        if len(img1.shape) == 2:
            img1 = np.tile(img1[...,None], (1, 1, 3))
            img2 = np.tile(img2[...,None], (1, 1, 3))
            img3 = np.tile(img3[..., None], (1, 1, 3))
        else:
            img1 = img1[..., :3]
            img2 = img2[..., :3]
            img3 = img3[..., :3]

        if self.augmentor is not None:
            if self.sparse and self.flowgt == 'two':
                valid = [valid1, valid2]
                flow = [flow1, flow2]
                img1, img2, img3, flow1, flow2, valid1, valid2 = self.augmentor(img1, img2, img3, flow, valid)
            elif self.sparse and self.flowgt == 'one':
                img1, img2, img3, flow, valid = self.augmentor(img1, img2, img3, [flow], [valid])
            else:
                img1, img2, img3, flow1, flow2 = self.augmentor(img1, img2, img3, flow1, flow2)

        img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
        img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
        img3 = torch.from_numpy(img3).permute(2, 0, 1).float()

        if self.sparse and self.flowgt == 'one':
            flow = torch.from_numpy(flow).permute(2, 0, 1).float()
        else:
            flow1 = torch.from_numpy(flow1).permute(2, 0, 1).float()
            flow2 = torch.from_numpy(flow2).permute(2, 0, 1).float()


        if self.occ_list is not None:
            return img1, img2, img3, flow1, flow2, occ1, occ2, self.occ_list[index]
        elif self.flowgt == 'two':
            if valid is not None:
                valid1 = torch.from_numpy(valid1)
                valid2 = torch.from_numpy(valid2)
            else:
                valid1 = (flow1[0].abs() < 1000) & (flow1[1].abs() < 1000)
                valid2 = (flow2[0].abs() < 1000) & (flow2[1].abs() < 1000)

            return img1, img2, img3, flow1, flow2, valid1.float(), valid2.float()

        else:
            valid = torch.from_numpy(valid)
            return img1, img2, img3, flow, valid.float()


    def __rmul__(self, v):
        self.flow_list = v * self.flow_list
        self.image_list = v * self.image_list
        return self
        
    def __len__(self):
        return len(self.image_list)
        

class MpiSintel(FlowDataset):
    def __init__(self, aug_params=None, split='training', root='datasets/Sintel', dstype='clean', occlusion=False):
        super(MpiSintel, self).__init__(aug_params)
        flow_root = osp.join(root, split, 'flow')
        image_root = osp.join(root, split, dstype)
        occ_root = osp.join(root, split, 'occlusions')

        self.occlusion = occlusion
        if self.occlusion:
            self.occ_list = []

        if split == 'test':
            self.is_test = True
        
        for scene in os.listdir(image_root):

            image_list = sorted(glob(osp.join(image_root, scene, '*.png')))

            if len(image_list)%2 == 0:
                for i in range(0, len(image_list)-2, 2):
                    self.image_list += [[image_list[i], image_list[i + 1], image_list[i + 2]]]
                    self.extra_info += [(scene, i)]  # scene and frame_id
                self.image_list += [[image_list[i+1], image_list[i+2], image_list[i + 3]]]
                self.extra_info += [(scene, i+1)]
            else:
                for i in range(0, len(image_list) - 2, 2):
                    self.image_list += [[image_list[i], image_list[i + 1], image_list[i+2]]]
                    self.extra_info += [(scene, i)]  # scene and frame_id

            if split != 'test':
                flow_list = sorted(glob(osp.join(flow_root, scene, '*.flo')))
                if len(flow_list)%2 == 0:
                    for i in range(0, len(flow_list),2):
                        self.flow_list += [ [flow_list[i], flow_list[i+1]] ]
                else:
                    for i in range(0, len(flow_list)-2, 2):
                        self.flow_list += [[flow_list[i], flow_list[i + 1]]]
                    self.flow_list += [[flow_list[-2], flow_list[-1]]]
                if self.occlusion:
                    occ_list = sorted(glob(osp.join(occ_root, scene, '*.png')))
                    if len(occ_list) % 2 == 0:
                        for i in range(0, len(occ_list), 2):
                            self.occ_list += [[occ_list[i], occ_list[i + 1]]]
                    else:
                        for i in range(0, len(occ_list) - 2, 2):
                            self.occ_list += [[occ_list[i], occ_list[i + 1]]]
                        self.occ_list += [[occ_list[-2], occ_list[-1]]]

#in-linear time direction
class FlyingThings3D(FlowDataset):
    def __init__(self, aug_params=None, root='datasets/FlyingThings3D', dstype='frames_cleanpass'):
        super(FlyingThings3D, self).__init__(aug_params)


        for cam in ['left']:
            for direction in ['into_future', 'into_past']:
                image_dirs = sorted(glob(osp.join(root, dstype, 'TRAIN/*/*')))
                image_dirs = sorted([osp.join(f, cam) for f in image_dirs])

                flow_dirs = sorted(glob(osp.join(root, 'optical_flow/TRAIN/*/*')))
                flow_dirs = sorted([osp.join(f, direction, cam) for f in flow_dirs])

                for idir, fdir in zip(image_dirs, flow_dirs):
                    images = sorted(glob(osp.join(idir, '*.png')))
                    flows = sorted(glob(osp.join(fdir, '*.pfm')))
                    for i in range(0, len(flows) - 2, 2):
                        if direction == 'into_future':
                            self.image_list += [[images[i], images[i+1], images[i+2]]]
                            self.flow_list += [[flows[i], flows[i+1]]]
                            if i == 6:
                                self.image_list += [[images[i+1], images[i+2], images[i+3]]]
                                self.flow_list += [[flows[i+1], flows[i+2]]]
                        elif direction == 'into_past':
                            self.image_list += [[images[i+2], images[i+1], images[i]]]
                            self.flow_list += [[flows[i+2], flows[i+1]]]
                            if i == 6:
                                self.image_list += [[images[i+3], images[i+2], images[i+1]]]
                                self.flow_list += [[flows[i+3], flows[i+2]]]

class VirtualKITTI2(FlowDataset):
    def __init__(self, aug_params=None, split='training', root='datasets/VirtualKITTI2'):
        super(VirtualKITTI2, self).__init__(aug_params, sparse='True')

        flow_root = osp.join(root, split, 'flowgt')
        image_root = osp.join(root, split, 'frames_rgb')

        for cam in ['Camera_0', 'Camera_1']:
            for direction in ['forward_flow', 'backward_flow']:

                for scene in sorted(os.listdir(image_root)):

                    image_dir = sorted(glob(osp.join(image_root, scene, '*')))
                    flow_dir = sorted(glob(osp.join(flow_root, direction, scene, '*')))

                    for idir, fdir in zip(image_dir, flow_dir):

                        images = sorted(glob(osp.join(idir, cam,'*.jpg')))
                        flows = sorted(glob(osp.join(fdir, cam, '*.png')))

                        for i in range(0, len(flows) - 2, 2):
                            if direction == 'forward_flow':
                                self.image_list += [[images[i], images[i + 1], images[i + 2]]]
                                self.flow_list += [[flows[i], flows[i + 1]]]
                            elif direction == 'backward_flow':
                                self.image_list += [[images[i+2], images[i+1], images[i]]]
                                self.flow_list += [[flows[i+1], flows[i]]]

class VirtualKITTI(FlowDataset):
    def __init__(self, aug_params=None, split = 'training', root='datasets/VirtualKITTI'):
        super(VirtualKITTI, self).__init__(aug_params, sparse='True')

        flow_root = osp.join(root, split, 'flowgt')
        image_root = osp.join(root, split, 'frames_rgb')
   
        for scene in sorted(os.listdir(image_root)):

            image_list = sorted(glob(osp.join(image_root, scene, '*')))
            flow_list = sorted(glob(osp.join(flow_root, scene, '*')))
            print(image_list)

            for idir, fdir in zip(image_list, flow_list):

                images = sorted(glob(osp.join(idir, '*.png')))
                flows =  sorted(glob(osp.join(fdir, '*.png')))

                if len(flows) % 2 == 0:
                    for i in range(0, len(flows) - 2, 2):
                        self.image_list += [[images[i], images[i + 1], images[i + 2]]]
                        self.flow_list += [[flows[i], flows[i+1]]]

                else:
                    for i in range(0, len(flows) - 2, 2):
                        self.image_list += [[images[i], images[i + 1], images[i + 2]]]
                        self.flow_list += [[flows[i], flows[i + 1]]]



class KITTI(FlowDataset):
    def __init__(self, aug_params=None, split='training', root='datasets/KITTI_MultiView'):
        super(KITTI, self).__init__(aug_params, sparse=True, flowgt = 'one')
        if split == 'testing':
            self.is_test = True

        root = osp.join(root, split)
        #images1 = sorted(glob(osp.join(root, 'image_2/*_10.png')))
        #images2 = sorted(glob(osp.join(root, 'image_2/*_11.png')))
        #images3 = sorted(glob(osp.join(root, 'image_2/*_12.png')))

        images1 = sorted(glob(osp.join(root, 'image_2/*_09.png')))
        images2 = sorted(glob(osp.join(root, 'image_2/*_10.png')))
        images3 = sorted(glob(osp.join(root, 'image_2/*_11.png')))

        for img1, img2, img3 in zip(images1, images2, images3):
            frame_id = img2.split('/')[-1]
            self.extra_info += [ [frame_id] ]
            self.image_list += [ [img1, img2, img3] ]

        if split == 'training':
            flow = sorted(glob(osp.join(root, 'flow_occ/*_10.png')))
            for flo1 in flow:
                self.flow_list += [[flo1]]

class Speckle(FlowDataset):
    def __init__(self, aug_params=None, split='training', root='datasets/Speckle'):
        super(Speckle, self).__init__(aug_params)
        if split == 'testing':
            self.is_test = True

        image_path = osp.join(root, split, 'Sequences')
        flow_path = osp.join(root, split, 'Flow')
        for scene in os.listdir(image_path):

            images = sorted(glob(osp.join(image_path, scene, '*.png')))
            flows = sorted(glob(osp.join(flow_path, scene, '*.flo')))

            self.image_list += [[images[0], images[1], images[2]]]
            self.flow_list += [[flows[0], flows[1]]]

            self.extra_info += [scene]

class Monkaa(FlowDataset):
    def __init__(self, aug_params=None, split='training', root='datasets/Monkaa', dstype='frames_cleanpass'):
        super(Monkaa, self).__init__(aug_params)

        for cam in ['left', 'right']:
            for direction in ['into_future', 'into_past']:
                image_dirs = sorted(glob(osp.join(root, split, dstype, '*')))
                image_dirs = sorted([osp.join(f, cam) for f in image_dirs])

                flow_dirs = sorted(glob(osp.join(root, split, 'optical_flow/*')))
                flow_dirs = sorted([osp.join(f, direction, cam) for f in flow_dirs])

                for idir, fdir in zip(image_dirs, flow_dirs):
                    images = sorted(glob(osp.join(idir, '*.png')) )
                    flows = sorted(glob(osp.join(fdir, '*.pfm')) )

                    if len(flows)%2==0:
                        for i in range(0, len(flows)-2, 2):
                            if direction == 'into_future':
                                self.image_list += [[images[i], images[i+1], images[i+2]]]
                                self.flow_list += [[flows[i], flows[i+1]]]
                            elif direction == 'into_past':
                                self.image_list += [[images[i+2], images[i+1], images[i]]]
                                self.flow_list += [[flows[i+2], flows[i+1]]]
                    else:
                        for i in range(0, len(flows)-1, 2):
                            if direction == 'into_future':
                                self.image_list += [[images[i], images[i + 1], images[i + 2]]]
                                self.flow_list += [[flows[i], flows[i+1]]]
                            elif direction == 'into_past':
                                self.image_list += [[images[i+2], images[i+1], images[i]]]
                                self.flow_list += [[flows[i+2], flows[i+1]]]


class HD1K(FlowDataset):
    def __init__(self, aug_params=None, root='datasets/HD1k'):
        super(HD1K, self).__init__(aug_params, sparse=True)

        seq_ix = 0
        while 1:
            flows = sorted(glob(os.path.join(root, 'hd1k_flow_gt', 'flow_occ/%06d_*.png' % seq_ix)))
            images = sorted(glob(os.path.join(root, 'hd1k_input', 'image_2/%06d_*.png' % seq_ix)))

            if len(flows) == 0:
                break

            elif len(flows) % 2 == 0:
                for i in range(0, len(flows)-2, 2):
                    self.flow_list += [[flows[i], flows[i+1]]]
                    self.image_list += [[images[i], images[i+1], images[i+2]]]
                self.flow_list += [[flows[-2], flows[-1]]]
                self.image_list += [[images[-3], images[-2], images[-1]]]
            else:
                for i in range(0, len(flows)-1, 2):
                    self.flow_list += [[flows[i], flows[i+1]]]
                    self.image_list += [[images[i], images[i + 1], images[i + 2]]]


            seq_ix += 1

class Retinal(FlowDataset):
    def __init__(self, aug_params=None, split='test', root='datasets/Retinal', dataset='DIGS'):

        super(Retinal, self).__init__(aug_params)
        image_root = osp.join(root, dataset)

        if split == 'test':
            self.is_test = True

        for scene in os.listdir(image_root):
            #Add all possible image extentions
            ext = ['*.png', '*.jpg', '*.PNG']
            image_list = []
            for i in range(len(ext)):
                image_list += sorted(glob(osp.join(image_root, scene, ext[i])))

            """ Original setup """
            self.image_list += [[image_list[0], image_list[1], image_list[2]]]
            """ First image, and last image duplicated twice """
            #self.image_list += [[image_list[0], image_list[2], image_list[2]]]

            self.extra_info += [scene]

# class Retinal2(FlowDataset):
#     def __init__(self, aug_params=None, split='test', root='datasets/Retinal', dataset='DIGS'):
#         if dataset == 'DIGS':
#             color = 'gray'
#             ext = '*.PNG'
#         elif dataset == 'LEGS':
#             color = 'rgb'
#             ext = '*.png'
#         elif dataset == 'Sclera':
#             color = 'gray'
#             ext = '*.jpg'
#         elif dataset == 'DIC':
#             color = 'gray'
#             ext = '*.png'
#
#         super(Retinal, self).__init__(aug_params, type=color)
#         image_root = osp.join(root, dataset)
#
#         if split == 'test':
#             self.is_test = True
#
#         for scene in os.listdir(image_root):
#
#             #images1 = sorted(glob(osp.join(root, scene, 'frame_0010.png')))
#             #images2 = sorted(glob(osp.join(root, scene, 'frame_0011.png')))
#             #images3 = sorted(glob(osp.join(root, scene, 'frame_0012.png')))
#
#             #self.image_list += [[images1, images2, images3]]
#
#             image_list = sorted(glob(osp.join(image_root, scene, ext)))
#             """ Original setup """
#             self.image_list += [[image_list[0], image_list[1], image_list[2]]]
#             """ First image, and last image duplicated twice """
#             #self.image_list += [[image_list[0], image_list[2], image_list[2]]]
#
#             self.extra_info += [scene]


def fetch_dataloader(args, TRAIN_DS='C+T+K+S+H'):
    """ Create the data loader for the corresponding trainign set """
    if args.stage == 'virtualKitti':
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.2, 'max_scale': 0.4, 'do_flip': False}
        #train_dataset = VirtualKITTI(aug_params, split='training')
        train_dataset = VirtualKITTI2(aug_params, split='training')

    elif args.stage == 'things':
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.4, 'max_scale': 0.8, 'do_flip': True}
        clean_dataset = FlyingThings3D(aug_params, dstype='frames_cleanpass')
        final_dataset = FlyingThings3D(aug_params, dstype='frames_finalpass')
        train_dataset = clean_dataset + final_dataset

    elif args.stage == 'chairs':
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.1, 'max_scale': 1.0, 'do_flip': True}
        train_dataset = FlyingChairs(aug_params, split='training')

    elif args.stage == 'monkaa':
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.4, 'max_scale': 0.8, 'do_flip': True}
        clean_dataset = Monkaa(aug_params, dstype='frames_cleanpass')
        final_dataset = Monkaa(aug_params, dstype='frames_finalpass')
        clean_val_dataset = Monkaa(aug_params, split='validation', dstype='frames_cleanpass')
        final_val_dataset = Monkaa(aug_params, split='validation', dstype='frames_finalpass')
        train_dataset = clean_dataset + final_dataset + clean_val_dataset + final_val_dataset

    elif args.stage == 'hd1k':
        hd1k = HD1K({'crop_size': args.image_size, 'min_scale': -0.5, 'max_scale': 0.2, 'do_flip': True})
        train_dataset = hd1k

    elif args.stage == 'sintel':
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.2, 'max_scale': 0.6, 'do_flip': True}
        things = FlyingThings3D(aug_params, dstype='frames_cleanpass')
        sintel_clean = MpiSintel(aug_params, split='training', dstype='clean')
        sintel_final = MpiSintel(aug_params, split='training', dstype='final')
        monkaa_tr = Monkaa(aug_params, dstype='frames_cleanpass')
        monkaa_val = Monkaa(aug_params, split='validation', dstype='frames_cleanpass')

        if TRAIN_DS == 'C+T+K+S+H':
            hd1k = HD1K({'crop_size': args.image_size, 'min_scale': -0.5, 'max_scale': 0.2, 'do_flip': True})
            vkitti = VirtualKITTI2(aug_params, split='validation')
            #train_dataset = 100*sintel_clean + 100*sintel_final + 5*hd1k + 2*monkaa_val + monkaa_tr + 5*vkitti
            train_dataset = 100*sintel_clean + 100*sintel_final + 5*hd1k + things + 4*vkitti
            #train_dataset = 200 * sintel_clean + 200 * sintel_final + 10*hd1k + things + 9*vkitti + monkaa_val + monkaa_tr

        elif TRAIN_DS == 'C+T+K/S':
            train_dataset = 100 * sintel_clean + 100 * sintel_final
    elif args.stage == 'speckle':
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.2, 'max_scale': 0.6, 'do_flip': True}

        sintel_clean = MpiSintel(aug_params, split='training', dstype='clean')
        sintel_final = MpiSintel(aug_params, split='training', dstype='final')

        speckle = Speckle(aug_params, split='training')
        print(len(sintel_final), len(sintel_clean), len(speckle) )
        train_dataset = speckle + sintel_final + sintel_clean

    elif args.stage == 'kitti':
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.2, 'max_scale': 0.4, 'do_flip': False, 'flow_gt':'one'}
        train_dataset = KITTI(aug_params, split='training')

    train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size,
        pin_memory=False, shuffle=True, num_workers=4, drop_last=True)

    print('Training with %d image triplets' % len(train_dataset))
    return train_loader

