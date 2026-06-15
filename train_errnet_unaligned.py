from os.path import join
from options.errnet.train_options import TrainOptions
from engine import Engine
from data.image_folder import read_fns
import torch
import torch.backends.cudnn as cudnn
import data.reflect_dataset as datasets
import util.util as util
import data

opt = TrainOptions().parse()

cudnn.benchmark = True

# processed datasets prepared by datasets/prepare_train_data.py and datasets/prepare_test_data.py
datadir = './datasets/processed_data'
raw_datadir = './datasets/raw_data'

datadir_syn = join(datadir, 'VOCdevkit/VOC2012/PNGImages')
datadir_real = join(datadir, 'real_train')
datadir_unaligned = join(raw_datadir, 'Dataset/DSLR/unaligned_train250')

train_dataset = datasets.CEILDataset(datadir_syn, read_fns('VOC2012_224_train_png.txt'), size=opt.max_dataset_size)
train_dataset_real = datasets.CEILTestDataset(datadir_real, enable_transforms=True)

train_dataset_unaligned = datasets.CEILTestDataset(datadir_unaligned, enable_transforms=True, flag={'unaligned':True}, size=None)

train_dataset_fusion = datasets.FusionDataset([train_dataset, train_dataset_unaligned, train_dataset_real], [0.25,0.5,0.25])


train_dataloader_fusion = datasets.DataLoader(
    train_dataset_fusion, batch_size=opt.batchSize, shuffle=not opt.serial_batches, 
    num_workers=opt.nThreads, pin_memory=True)


engine = Engine(opt)
"""Main Loop"""
def set_learning_rate(lr):
    for optimizer in engine.model.optimizers:
        util.set_opt_param(optimizer, 'lr', lr)


def has_nan_weights(model):
    for module in (model.net_i, model.net_rd):
        if module is None:
            continue
        for param in module.parameters():
            if not torch.isfinite(param).all():
                return True
    return False


set_learning_rate(opt.lr)
while engine.epoch < 80:
    if engine.epoch == 70:
        set_learning_rate(opt.lr * 0.2)

    engine.train(train_dataloader_fusion)

    if has_nan_weights(engine.model):
        print('[ERROR] NaN/Inf in model weights after epoch %d, stopping.' % engine.epoch)
        break
