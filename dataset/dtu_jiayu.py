# Dataloader for the DTU dataset in Yaoyao's format.
# The code is borrowed from the following github repository:
#       https://github.com/JiayuYANG/CVP-MVSNet

from dataset.utils import *
from dataset.dataPaths import *
from torch.utils.data import Dataset
import numpy as np
import os
import sys
from PIL import Image
# from base_data_loader import BaseDataLoader
# from .kitti_dataset import *
# For debug:
import matplotlib.pyplot as plt
# import pdb


class MVSDataset(Dataset):
    def __init__(self, args, logger=None):
        # Initializing the dataloader
        super(MVSDataset, self).__init__()
        
        # Parse input
        self.args = args
        self.data_root = self.args.dataset_root
        self.scan_list_file = getScanListFile(self.data_root,self.args.mode)
        self.pair_list_file = getPairListFile(self.data_root,self.args.mode)
        self.logger = logger
        if logger==None:
            import logging
            self.logger = logging.getLogger()
            self.logger.setLevel(logging.DEBUG)
            formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
            consoleHandler = logging.StreamHandler(sys.stdout)
            consoleHandler.setFormatter(formatter)
            self.logger.addHandler(consoleHandler)
            self.logger.info("File logger not configured, only writing logs to stdout.")
        self.logger.info("Initiating dataloader for our pre-processed DTU dataset.")
        self.logger.info("Using dataset:"+self.data_root+self.args.mode+"/")

        self.metas = self.build_list(self.args.mode)
        self.logger.info("Dataloader initialized.")

    def build_list(self,mode):
        # Build the item meta list
        metas = []

        # Read scan list
        scan_list = readScanList(self.scan_list_file,self.args.mode, self.logger)

        # Read pairs list
        for scan in scan_list:
            with open(self.pair_list_file) as f:
                num_viewpoint = int(f.readline())
                # viewpoints (49)
                for view_idx in range(num_viewpoint):
                    ref_view = int(f.readline().rstrip())
                    src_views = [int(x) for x in f.readline().rstrip().split()[1::2]]
                    # light conditions 0-6
                    if mode == "train":
                        for light_idx in range(7):
                            metas.append((scan, ref_view, src_views, light_idx))
                    elif mode == "test":
                        metas.append((scan, ref_view, src_views, 3))
        self.logger.info("Done. metas:"+str(len(metas)))
        return metas

    def __len__(self):
        return len(self.metas)

    def __getitem__(self, idx):
        meta = self.metas[idx]
        scan, ref_view, src_views, light_idx = meta

        assert self.args.nsrc <= len(src_views)

        self.logger.debug("Getting Item:\nscan:"+str(scan)+"\nref_view:"+str(ref_view)+"\nsrc_view:"+str(src_views)+"\nlight_idx"+str(light_idx))

        ref_img = [] 
        src_imgs = [] 
        ref_depths = [] 
        ref_depth_mask = [] 
        ref_intrinsics = [] 
        src_intrinsics = [] 
        ref_extrinsics = [] 
        src_extrinsics = [] 
        depth_min = [] 
        depth_max = [] 

        ## 1. Read images
        # ref image
        ref_img_file = getImageFile(self.data_root,self.args.mode,scan,ref_view,light_idx)
        ref_img = read_img(ref_img_file)

        # src image(s)
        for i in range(self.args.nsrc):
            src_img_file = getImageFile(self.data_root,self.args.mode,scan,src_views[i],light_idx)
            src_img = read_img(src_img_file)

            src_imgs.append(src_img)

        ## 2. Read camera parameters
        cam_file = getCameraFile(self.data_root,self.args.mode,ref_view)
        ref_intrinsics, ref_extrinsics, depth_min, depth_max = read_cam_file(cam_file)
        for i in range(self.args.nsrc):
            cam_file = getCameraFile(self.data_root,self.args.mode,src_views[i])
            intrinsics, extrinsics, depth_min_tmp, depth_max_tmp = read_cam_file(cam_file)
            src_intrinsics.append(intrinsics)
            src_extrinsics.append(extrinsics)

        ## 3. Read Depth Maps
        if self.args.mode == "train":
            imgsize = self.args.imgsize
            nscale = self.args.nscale

            # Read depth map of same size as input image first.
            depth_file = getDepthFile(self.data_root,self.args.mode,scan,ref_view)
            ref_depth = read_depth(depth_file)

            depth_frame_size = (ref_depth.shape[0],ref_depth.shape[1])
            frame = np.zeros(depth_frame_size)
            frame[:ref_depth.shape[0],:ref_depth.shape[1]] = ref_depth
            ref_depths.append(frame)

            # Downsample the depth for each scale.
            ref_depth = Image.fromarray(ref_depth)
            original_size = np.array(ref_depth.size).astype(int)


            for scale in range(1,nscale):
                new_size = (original_size/(2**scale)).astype(int)
                down_depth = ref_depth.resize((new_size),Image.BICUBIC)
                frame = np.zeros(depth_frame_size)
                down_np_depth = np.array(down_depth)
                frame[:down_np_depth.shape[0],:down_np_depth.shape[1]] = down_np_depth
                ref_depths.append(frame)
        # Orgnize output and return
        sample = {}
        sample["ref_img"] = np.moveaxis(np.array(ref_img),2,0) # (3 128 160)
        sample["src_imgs"] = np.moveaxis(np.array(src_imgs),3,1) #(2, 3, 128, 160)
        sample["ref_intrinsics"] = np.array(ref_intrinsics) # (3, 3)
        sample["src_intrinsics"] = np.array(src_intrinsics) # (2, 3, 3)
        sample["ref_extrinsics"] = np.array(ref_extrinsics) # (4, 4)
        sample["src_extrinsics"] = np.array(src_extrinsics) # (2, 4, 4)
        sample["depth_min"] = depth_min
        sample["depth_max"] = depth_max
        print(sample["ref_intrinsics"])
        sys.exit()

        # print(sample["src_imgs"].shape)
        # print(sample["ref_intrinsics"].shape)
        # print(sample["src_intrinsics"].shape)
        # print(sample["ref_extrinsics"].shape)
        # print(sample["src_extrinsics"].shape)
        #print(sample)



        if self.args.mode == "train":
            sample["ref_depths"] = np.array(ref_depths,dtype=float) # (2, 128, 160)
            sample["ref_depth_mask"] = np.array(ref_depth_mask)
        elif self.args.mode == "test":
            sample["filename"] = scan + '/{}/' + '{:0>8}'.format(ref_view) + "{}"

        return sample

# class KittiOdometryDataloader(BaseDataLoader):
#
#     def __init__(self, batch_size=1, shuffle=True, validation_split=0.0, num_workers=4, **kwargs):
#         self.dataset = KittiOdometryDataset(**kwargs)
#         super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


