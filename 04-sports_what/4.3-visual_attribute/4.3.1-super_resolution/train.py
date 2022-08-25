import datetime
import logging
import math
import time
from os import path as osp
from data.realesrgan_dataset import RealESRGANDataset
from data.realesrgan_paired_dataset import RealESRGANPairedDataset
import paddle
from tqdm import tqdm
from paddle.io import DataLoader
import paddle.distributed as dist

from data.dataset import build_dataloader  # , build_dataset
from data.data_sampler import EnlargedSampler
from models.build_model import build_model
from utils.logger import get_root_logger, MessageLogger, AvgTimer
from utils.options import copy_opt_file, dict2str, parse_options
from utils.scandir import scandir
from utils.misc import check_resume, make_exp_dirs, mkdir_and_rename, get_time_str
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--yml_path', type=str, default='options/train_realesrgan_x4plus.yml', help='yml file')
args = parser.parse_args()


def create_train_val_dataloader(opt, logger):
    # create train and val dataloaders
    train_loader, val_loaders = None, []
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            # train_set = build_dataset(dataset_opt)
            train_set = RealESRGANDataset(dataset_opt)
            batch_sampler = paddle.io.DistributedBatchSampler(
                train_set, batch_size=dataset_opt['batch_size_per_gpu'],
                shuffle=dataset_opt['use_shuffle'], drop_last=False)

            train_loader = DataLoader(train_set,
                                      batch_sampler=batch_sampler,
                                      num_workers=dataset_opt['num_worker_per_gpu'],
                                      return_list=True)

            num_iter_per_epoch = math.ceil(
                len(train_set) / (dataset_opt['batch_size_per_gpu'] * opt['nranks']))
            total_iters = int(opt['train']['total_iter'])
            total_epochs = math.ceil(total_iters / (num_iter_per_epoch))
            logger.info('Training statistics:'
                        f'\n\tNumber of train images: {len(train_set)}'
                        f'\n\tBatch size per gpu: {dataset_opt["batch_size_per_gpu"]}'
                        f'\n\tWorld size (gpu number): {opt["world_size"]}'
                        f'\n\tRequire iter number per epoch: {num_iter_per_epoch}'
                        f'\n\tTotal epochs: {total_epochs}; iters: {total_iters}.')

        elif phase.split('_')[0] == 'val':
            val_set = RealESRGANPairedDataset(dataset_opt)
            val_loader = build_dataloader(
                val_set, dataset_opt, num_gpu=opt['num_gpu'], dist=opt['dist'], sampler=None, seed=opt['manual_seed'])
            logger.info(f'Number of val images/folders in {dataset_opt["name"]}: {len(val_set)}')
            val_loaders.append(val_loader)
        else:
            raise ValueError(f'Dataset phase {phase} is not recognized.')

    return train_loader, val_loaders, total_epochs, total_iters


def load_resume_state(opt):
    resume_state_path = None
    if opt['auto_resume']:
        state_path = osp.join('experiments', opt['name'], 'training_states')
        if osp.isdir(state_path):
            states = list(scandir(state_path, suffix='state', recursive=False, full_path=False))
            if len(states) != 0:
                states = [float(v.split('.state')[0]) for v in states]
                resume_state_path = osp.join(state_path, f'{max(states):.0f}.state')
                opt['path']['resume_state'] = resume_state_path
    else:
        if opt['path'].get('resume_state'):
            resume_state_path = opt['path']['resume_state']

    if resume_state_path is None:
        resume_state = None
    else:
        resume_state = paddle.load(resume_state_path)
        check_resume(opt, resume_state['iter'])
    return resume_state


def train_pipeline(root_path, yml_path):
    # ----------------------------------------
    # parallel
    # ----------------------------------------
    dist.init_parallel_env()
    nranks = paddle.distributed.ParallelEnv().nranks
    local_rank = paddle.distributed.ParallelEnv().local_rank
    print(nranks)

    # parse options, set distributed setting, set ramdom seed
    opt, args = parse_options(root_path, yml_path, is_train=True)
    opt['root_path'] = root_path

    # load resume states if necessary
    resume_state = load_resume_state(opt)
    # mkdir for experiments and logger
    if resume_state is None:
        if local_rank == 0:
            make_exp_dirs(opt)
            if opt['logger'].get('use_tb_logger') and 'debug' not in opt['name'] and opt['rank'] == 0:
                mkdir_and_rename(osp.join(opt['root_path'], 'tb_logger', opt['name']))

    # copy the yml file to the experiment root
    copy_opt_file(args.opt, opt['path']['experiments_root'])

    # WARNING: should not use get_root_logger in the above codes, including the called functions
    # Otherwise the logger will not be properly initialized
    log_file = osp.join(opt['path']['log'], f"train_{opt['name']}_{get_time_str()}.log")
    logger = get_root_logger(logger_name='basicsr', log_level=logging.INFO, log_file=log_file)
    logger.info(dict2str(opt))
    # tb_logger = init_tb_loggers(opt)

    opt['nranks'] = nranks

    # create train and validation dataloaders
    result = create_train_val_dataloader(opt, logger)
    train_loader, val_loaders, total_epochs, total_iters = result

    # create model
    model = build_model(opt)

    if resume_state:  # resume training
        model.resume_training(resume_state)  # handle optimizers and schedulers
        logger.info(f"Resuming training from epoch: {resume_state['epoch']}, " f"iter: {resume_state['iter']}.")
        start_epoch = resume_state['epoch']
        current_iter = resume_state['iter']
    else:
        start_epoch = 0
        current_iter = 0

    # create message logger (formatted outputs)
    msg_logger = MessageLogger(opt, current_iter)

    # training
    logger.info(f'Start training from epoch: {start_epoch}, iter: {current_iter}')
    data_timer, iter_timer = AvgTimer(), AvgTimer()
    start_time = time.time()

    for epoch in range(start_epoch, total_epochs + 1):

        # while train_data is not None:
        for _, (train_data) in tqdm(enumerate(train_loader), total=len(train_loader)):
            # print(i)
            data_timer.record()

            current_iter += 1
            if current_iter > total_iters:
                break
            # update learning rate
            model.update_learning_rate(current_iter, warmup_iter=opt['train'].get('warmup_iter', -1))
            # training
            model.feed_data(train_data)
            model.optimize_parameters(current_iter)
            iter_timer.record()

            if current_iter == 1:
                # reset start time in msg_logger for more accurate eta_time
                # not work in resume mode
                if local_rank == 0:
                    msg_logger.reset_start_time()
            # log
            if current_iter % opt['logger']['print_freq'] == 0:
                if local_rank == 0:
                    log_vars = {'epoch': epoch, 'iter': current_iter}
                    log_vars.update({'lrs': model.get_current_learning_rate()})
                    log_vars.update({'time': iter_timer.get_avg_time(), 'data_time': data_timer.get_avg_time()})
                    log_vars.update(model.get_current_log())
                    msg_logger(log_vars)

            # save models and training states
            if current_iter % opt['logger']['save_checkpoint_freq'] == 0:
                logger.info('Saving models and training states.')
                model.save(epoch, current_iter)

            # validation
            if opt.get('val') is not None and (current_iter % opt['val']['val_freq'] == 0):
                if len(val_loaders) > 1:
                    logger.warning('Multiple validation datasets are *only* supported by SRModel.')
                for val_loader in val_loaders:
                    model.validation(val_loader, current_iter, opt['val']['save_img'])

            data_timer.start()
            iter_timer.start()
        # end of iter

    # end of epoch

    consumed_time = str(datetime.timedelta(seconds=int(time.time() - start_time)))
    logger.info(f'End of training. Time consumed: {consumed_time}')
    logger.info('Save the latest model.')
    model.save(epoch=-1, current_iter=-1)  # -1 stands for the latest
    if opt.get('val') is not None:
        for val_loader in val_loaders:
            model.validation(val_loader, current_iter, opt['val']['save_img'])
    # if tb_logger:
    #     tb_logger.close()


if __name__ == '__main__':
    yml_path = args.yml_path
    root_path = './'
    train_pipeline(root_path, yml_path)
