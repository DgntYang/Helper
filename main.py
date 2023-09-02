import csv
import os
import os.path as osp
import shutil
import time

import torch


# used to create some related excel files
fieldnames = [
    'epoch', 'rmse', 'photo', 'mae', 'irmse', 'imae', 'mse', 'absrel', 'lg10',
    'silog', 'squared_rel', 'delta1', 'delta2', 'delta3', 'data_time',
    'gpu_time'
]


# to backup some important codes
ignore_hidden = shutil.ignore_patterns(".", "..", ".git*", "*pycache*",
                                       "*build", "*.fuse*", "*_drive_*")

def backup_source_code(backup_directory):
    if os.path.exists(backup_directory):
        shutil.rmtree(backup_directory)
    shutil.copytree('.', backup_directory, ignore=ignore_hidden)


# adjust learning rate
def adjust_learning_rate(lr_init, optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 5 epochs"""
    lr = lr_init * (0.1**(epoch // 5))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

# save my checkpoint to related output_directory
def save_checkpoint(state, is_best, epoch, output_directory):
    checkpoint_filename = os.path.join(output_directory,
                                       'checkpoint-' + str(epoch) + '.pth.tar')
    torch.save(state, checkpoint_filename)
    if is_best:
        best_filename = os.path.join(output_directory, 'model_best.pth.tar')
        shutil.copyfile(checkpoint_filename, best_filename)
    if epoch > 0:
        prev_checkpoint_filename = os.path.join(
            output_directory, 'checkpoint-' + str(epoch - 1) + '.pth.tar')
        if os.path.exists(prev_checkpoint_filename):
            os.remove(prev_checkpoint_filename)

# make folder's name according to args/cfg
def get_folder_name(args):
    current_time = time.strftime('%m-%d-%H-%M')
    return os.path.join(args.result,
        'M={}.t={}.b={}.s={}.lr={}.e={}.lf={}.dn={}.wd={}'.
        format(args.model_resume, current_time, args.batch_size, args.seed, args.lr, args.epochs, args.loss_function, args.denoise,args.weight_decay))










