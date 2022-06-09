# Dataloader for the ETH3D dataset in Yaoyao's format.
# Note: This file modified based code from the following projects.
#       https://github.com/JiayuYANG/CVP-MVSNet



from dataset.utils import *
from dataset.dataPaths import *
from torch.utils.data import Dataset
import numpy as np
import os
import sys
from PIL import Image
import math

# For debug:
import matplotlib.pyplot as plt
# import pdb


class ETHDataset(Dataset):
    def __init__(self, args, logger=None):
        # Initializing the dataloader
        super(ETHDataset, self).__init__()
        
        # Parse input
        self.args = args
        self.data_root = self.args.dataset_root
        self.scan_list_file = getScanListFile(self.data_root,self.args.mode)
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
            # print(os.path.join(self.data_root,scan))
            pair_list_file = getPairListFile(os.path.join(self.data_root, scan)+'/', self.args.mode)
            with open(pair_list_file) as f:
                num_viewpoint = int(f.readline())
                for view_idx in range(num_viewpoint):
                    ref_view = int(f.readline().rstrip())
                    src_views = [int(x) for x in f.readline().rstrip().split()[1::2]]
                    # light conditions 0-6

                    metas.append((scan, ref_view, src_views, 3))

        self.logger.info("Done. metas:"+str(len(metas)))
        return metas

    def __len__(self):
        return len(self.metas)

    def __getitem__(self, idx):
        meta = self.metas[idx]
        scan, ref_view, src_views, light_idx = meta
        imgsize = self.args.imgsize
        assert self.args.nsrc <= len(src_views)
        if self.args.mode == "train":
            n_size = (448, 240)
        else:
            n_size = (896, 480)
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
        ref_img_file = getImage(self.data_root,self.args.mode,scan,ref_view)
        ref_img = read_img(ref_img_file,n_size)#[:, :, :1]
        original_size = Image.open(ref_img_file).size
        # src image(s)
        for i in range(self.args.nsrc):
            src_img_file = getImage(self.data_root,self.args.mode,scan,src_views[i])
            src_img = read_img(src_img_file,n_size)#[:, :, :1]
            src_imgs.append(src_img)

        ## 2. Read camera parameters
        cam_file = getCameraFile(os.path.join(self.data_root,scan)+'/',self.args.mode,ref_view)
        ref_intrinsics, ref_extrinsics, depth_min, depth_max = read_cam_file(cam_file)

        ref_intrinsics[0] *= ref_img.shape[1] / original_size[0]
        ref_intrinsics[1] *= ref_img.shape[0] / original_size[1]

        for i in range(self.args.nsrc):
            cam_file = getCameraFile(os.path.join(self.data_root,scan)+'/',self.args.mode,src_views[i])
            intrinsics, extrinsics, depth_min_tmp, depth_max_tmp = read_cam_file(cam_file)
            intrinsics[0] *= ref_img.shape[1] / original_size[0]
            intrinsics[1] *= ref_img.shape[0] / original_size[1]
            src_intrinsics.append(intrinsics)
            src_extrinsics.append(extrinsics)

        ## 3. Read Depth Maps
        if self.args.mode == "train" or self.args.mode == "val":

            nscale = self.args.nscale

            # Read depth map of same size as input image first.
            depth_file = getDepth(self.data_root,self.args.mode,scan,ref_view)
            ref_depth = read_depth(depth_file)

            ref_depth = Image.fromarray(ref_depth)
            ref_depth = ref_depth.resize(n_size, resample=Image.NEAREST)
            ref_depth=np.array(ref_depth)
            depth_frame_size = (ref_depth.shape[0],ref_depth.shape[1])
            frame = np.zeros(depth_frame_size)
            frame[:ref_depth.shape[0],:ref_depth.shape[1]] = ref_depth
            ref_depths.append(frame)

            # Downsample the depth for each scale.
            ref_depth = Image.fromarray(ref_depth)
            original_size = np.array(ref_depth.size).astype(int)


            for scale in range(1,nscale):
                new_size = (original_size/(2**scale)).astype(int)
                down_depth = ref_depth.resize((new_size),resample=Image.NEAREST)
                frame = np.zeros(depth_frame_size)
                down_np_depth = np.array(down_depth)
                frame[:down_np_depth.shape[0],:down_np_depth.shape[1]] = down_np_depth
                ref_depths.append(frame)
        # Orgnize output and return
        sample = {}
        sample["ref_img"] = np.moveaxis(np.array(ref_img), 2, 0)  # (3 128 160)
        sample["src_imgs"] = np.moveaxis(np.array(src_imgs), 3, 1) #(2, 3, 128, 160)
        sample["ref_intrinsics"] = np.array(ref_intrinsics) # (3, 3)
        sample["src_intrinsics"] = np.array(src_intrinsics) # (2, 3, 3)
        sample["ref_extrinsics"] = np.array(ref_extrinsics) # (4, 4)
        sample["src_extrinsics"] = np.array(src_extrinsics) # (2, 4, 4)
        sample["depth_min"] = depth_min
        sample["depth_max"] = depth_max
        sample["img_path"] = str(scan)+str(ref_view).zfill(6)




        if self.args.mode == "train":
            sample["ref_depths"] = np.array(ref_depths,dtype=float) # (2, 128, 160)
            sample["ref_depth_mask"] = np.array(ref_depth_mask)
            sample["depth_min"] = math.floor(depth_min)
            sample["depth_max"] = math.ceil(depth_max)
        elif self.args.mode == "val":
            sample["ref_depths"] = np.array(ref_depths, dtype=float)  # (2, 128, 160)
            sample["ref_depth_mask"] = np.array(ref_depth_mask)
            sample["filename"] = scan + '/{}/' + '{:0>8}'.format(ref_view) + "{}"

        elif self.args.mode == "test":
            sample["filename"] = scan + '/{}/' + '{:0>8}'.format(ref_view) + "{}"

        return sample

