# Validate CVP-MVSNet
# Note: This file modified the code from the following projects.
#       https://github.com/JiayuYANG/CVP-MVSNet

import os,sys,time,logging,argparse,datetime,re
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import dtu_jiayu
from dataset import eth
from models import net
from models.modules import *
from utils import *
from PIL import Image
from argsParser import getArgsParser,checkArgs
import torch.utils
import torch.utils.checkpoint
import errno
# Debug import
# import pdb
import matplotlib.pyplot as plt



# Arg parser
parser = getArgsParser()
args = parser.parse_args()
assert args.mode == "val"
checkArgs(args)


torch.backends.cudnn.benchmark=True

# Check checkpoint directory
if not os.path.exists(args.logckptdir+args.info.replace(" ","_")):
    try:
        os.makedirs(args.logckptdir+args.info.replace(" ","_"))
    except OSError as exc: # Guard against race condition
        if exc.errno != errno.EEXIST:
            raise

# logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)
curTime = time.strftime('%Y%m%d-%H%M', time.localtime(time.time()))
log_path = args.loggingdir+args.info.replace(" ","_")+"/"
if not os.path.isdir(args.loggingdir):
    os.mkdir(args.loggingdir)
if not os.path.isdir(log_path):
    os.mkdir(log_path)
log_name = log_path + curTime + '.log'
logfile = log_name
formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
fileHandler = logging.FileHandler(logfile, mode='a')
fileHandler.setFormatter(formatter)
logger.addHandler(fileHandler)
consoleHandler = logging.StreamHandler(sys.stdout)
consoleHandler.setFormatter(formatter)
logger.addHandler(consoleHandler)
logger.info("Logger initialized.")
logger.info("Writing logs to file:"+logfile)

settings_str = "All settings:\n"
line_width = 30
for k,v in vars(args).items(): 
    settings_str += '{0}: {1}\n'.format(k,v)
logger.info(settings_str)

# Dataset
if args.dataloader == 'eth':
    val_dataset = eth.ETHDataset(args, logger)
else:
    val_dataset = dtu_jiayu.MVSDataset(args, logger)
val_loader = DataLoader(val_dataset, args.batch_size, shuffle=args.eval_shuffle, num_workers=8, drop_last=True)

# Network
model = net.network(args)
logger.info(">>> total params: {:.2f}M".format(sum(p.numel() for p in model.parameters()) / 1000000.0))
# model = nn.DataParallel(model)  
model.cuda()

if args.loss_function == "sl1":
    logger.info("Using smoothed L1 loss")
    model_loss = net.sL1_loss
else: # MSE
    logger.info("Using MSE loss")
    model_loss = net.MSE_loss



logger.info('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

def save_pfm(filename, image, scale=1):

    if not os.path.exists(os.path.dirname(filename)):
        try:
            os.makedirs(os.path.dirname(filename))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

    file = open(filename, "wb")
    color = None

    image = np.flipud(image)

    if image.dtype.name != 'float32':
        raise Exception('Image dtype must be float32.')

    if len(image.shape) == 3 and image.shape[2] == 3:  # color image
        color = True
    elif len(image.shape) == 2 or len(image.shape) == 3 and image.shape[2] == 1:  # greyscale
        color = False
    else:
        raise Exception('Image must have H x W x 3, H x W x 1 or H x W dimensions.')

    file.write('PF\n'.encode('utf-8') if color else 'Pf\n'.encode('utf-8'))
    file.write('{} {}\n'.format(image.shape[1], image.shape[0]).encode('utf-8'))

    endian = image.dtype.byteorder

    if endian == '<' or endian == '=' and sys.byteorder == 'little':
        scale = -scale

    file.write(('%f\n' % scale).encode('utf-8'))

    image.tofile(file)
    file.close()


def write_depth_img(filename,depth):

    if not os.path.exists(os.path.dirname(filename)):
        try:
            os.makedirs(os.path.dirname(filename))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
    # save the depth map
    plt.figure()
    plt.imshow(depth)

    # plt.colorbar()
    plt.savefig(filename)
    plt.close()
    # depth[depth <= 0.0] = 0.0
    # plt.imsave(filename, depth.squeeze())
    return 1
# main function
def validate():

    val_loss= []
    if (args.mode == "val" and not args.loadckpt):
        sw_path = args.logckptdir
        saved_models = [fn for fn in os.listdir(sw_path) if fn.endswith(".ckpt")]
        saved_models = sorted(saved_models, key=lambda x: int(x.split('_')[-1].split('.')[0]))
        args.epochs = len(saved_models)
    elif args.loadckpt:
        # load checkpoint file specified by args.loadckpt
        logger.info("loading model {}".format(args.loadckpt))
        state_dict = torch.load(args.loadckpt)
        model.load_state_dict(state_dict['model'])
        args.epochs = 1

    for epoch_idx in range(0, args.epochs):
            if args.loadckpt:
                pass
            else:
                loadckpt = os.path.join(sw_path, saved_models[epoch_idx])
                logger.info("Resuming " + loadckpt)
                state_dict = torch.load(loadckpt)
                model.load_state_dict(state_dict['model'])

            this_loss = []
            with torch.no_grad():
                for batch_idx, sample in enumerate(val_loader):

                    start_time = time.time()
                    loss = val_sample(sample)
                    this_loss.append(loss)
                    logger.info(
                        'Epoch {}/{}, Iter {}/{}, val loss = {:.3f},time = {:.3f}'.format(epoch_idx, args.epochs, batch_idx,
                                                                                      len(val_loader), loss,time.time() - start_time))
                val_loss.append(np.mean(this_loss))
                print(np.mean(this_loss))
                print(val_loss)
    print(val_loss)





def val_sample(sample, detailed_summary=False):



    sample_cuda = tocuda(sample)
    ref_depths = sample_cuda["ref_depths"]

    outputs = model(\
    sample_cuda["ref_img"].float(), \
    sample_cuda["src_imgs"].float(), \
    sample_cuda["ref_intrinsics"], \
    sample_cuda["src_intrinsics"], \
    sample_cuda["ref_extrinsics"], \
    sample_cuda["src_extrinsics"], \
    sample_cuda["depth_min"], \
    sample_cuda["depth_max"],sample_cuda["img_path"])

    depth_est_list = outputs["depth_est_list"]
    dHeight = ref_depths.shape[2]
    dWidth = ref_depths.shape[3]
    loss = []
    for i in range(0,args.nscale):

        depth_gt = ref_depths[:,i,:int(dHeight/2**i),:int(dWidth/2**i)]
        mask = depth_gt > 0.384
        loss.append(model_loss(depth_est_list[i], depth_gt.float(), mask))
    loss = sum(loss)

    depth_est = depth_est_list[0].data.cpu().numpy()
    prob_confidence = outputs["prob_confidence"].data.cpu().numpy()
    del sample_cuda
    filenames = sample["filename"]


    # # save depth maps and confidence maps
    for filename, est_depth, photometric_confidence in zip(filenames, depth_est, prob_confidence):
        depth_filename = os.path.join(args.outdir, filename.format('depth_est', '.pfm'))
        confidence_filename = os.path.join(args.outdir, filename.format('confidence', '.pfm'))
        os.makedirs(depth_filename.rsplit('/', 1)[0], exist_ok=True)
        os.makedirs(confidence_filename.rsplit('/', 1)[0], exist_ok=True)
        # save depth maps
        save_pfm(depth_filename, est_depth)

        # Save prob maps
        save_pfm(confidence_filename, photometric_confidence)


    return loss.data.cpu().item()


if __name__ == '__main__':
    if args.mode == "val":
        validate()
