import os
import shutil
import random
from tqdm import tqdm

import numpy as np

import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader

from tensorboardX import SummaryWriter

from graph.model.model import Model
from graph.model.regressor import Regressor
from graph.loss.loss import BCELoss, MSELoss
from data.dataset import Dataset

from utils.metrics import AverageMeter
from utils.train_utils import free, set_logger, count_model_prameters, get_lr


cudnn.benchmark = True


class Box(object):
    def __init__(self, config):
        self.config = config
        self.flag_gan = False
        self.train_count = 0
        self.best_val_loss = 9999.

        self.pretraining_step_size = self.config.pretraining_step_size
        self.batch_size = self.config.batch_size

        self.logger = set_logger('train_epoch.log')

        # define dataloader
        self.dataset = Dataset(self.config, 'train')
        self.dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False, num_workers=2,
                                     pin_memory=self.config.pin_memory, collate_fn=self.collate_function)

        self.val_set = Dataset(self.config, 'val')
        self.val_loader = DataLoader(self.val_set, batch_size=self.batch_size, shuffle=False, num_workers=1,
                                     pin_memory=self.config.pin_memory, collate_fn=self.collate_function)

        # define models
        self.reg = Regressor().cuda()

        # define loss
        self.mse = MSELoss().cuda()

        # define lr
        self.lr = self.config.learning_rate

        # define optimizer
        self.opt = torch.optim.Adam(self.reg.parameters(), lr=self.lr, eps=1e-6)

        # define optimize scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.opt, mode='min', factor=0.8, cooldown=7)

        # initialize train counter
        self.epoch = 0
        self.total_iter = (len(self.dataset) + self.config.batch_size - 1) // self.config.batch_size
        self.val_iter = (len(self.val_set) + self.config.batch_size - 1) // self.config.batch_size

        self.manual_seed = random.randint(10000, 99999)

        torch.manual_seed(self.manual_seed)
        torch.cuda.manual_seed_all(self.manual_seed)
        random.seed(self.manual_seed)

        # parallel setting
        gpu_list = list(range(self.config.gpu_cnt))
        self.reg = nn.DataParallel(self.reg, device_ids=gpu_list)

        # Model Loading from the latest checkpoint if not found start from scratch.
        self.load_checkpoint(self.config.checkpoint_file)

        # Summary Writer
        self.summary_writer = SummaryWriter(log_dir=os.path.join(self.config.root_path, self.config.summary_dir),
                                            comment='LayoutNet')
        self.print_train_info()

    def print_train_info(self):
        print('seed: ', self.manual_seed)
        print('Number of model parameters: {}'.format(count_model_prameters(self.reg)))

    def collate_function(self, samples):
        data = dict()

        data['corner'] = torch.from_numpy(np.array([sample['corner'] for sample in samples]))
        data['edge'] = torch.from_numpy(np.array([sample['edge'] for sample in samples]))
        data['box'] = torch.from_numpy(np.array([sample['box'] for sample in samples]))
        
        return data

    def load_checkpoint(self, file_name):
        filename = os.path.join(self.config.root_path, self.config.checkpoint_dir, 'reg_' + file_name)
        try:
            print('Loading checkpoint {}'.format(filename))
            checkpoint = torch.load(filename)

            self.reg.load_state_dict(checkpoint['reg_state_dict'])

        except OSError as e:
            print('No checkpoint exists from {}. Skipping...'.format(self.config.checkpoint_dir))
            print('**First time to train**')

    def save_checkpoint(self):
        tmp_name = os.path.join(self.config.root_path, self.config.checkpoint_dir,
                                'reg_checkpoint_{}.pth.tar'.format(self.epoch))

        state = {
            'reg_state_dict': self.reg.state_dict(),
        }

        torch.save(state, tmp_name)
        shutil.copyfile(tmp_name, os.path.join(self.config.root_path, self.config.checkpoint_dir,
                                               ('reg_' + self.config.checkpoint_file)))

    def run(self):
        try:
            self.train()

        except KeyboardInterrupt:
            print('You have entered CTRL+C.. Wait to finalize')

    def train(self):
        while self.epoch < self.config.epoch:
            self.epoch += 1
            self.train_by_epoch()
            self.validate_by_epoch()
                
    def train_by_epoch(self):
        tqdm_batch = tqdm(self.dataloader, total=self.total_iter, desc='epoch-{}'.format(self.epoch))

        avg_loss = AverageMeter()
        for curr_it, data in enumerate(tqdm_batch):
            self.reg.train()

            edge = data['edge'].float().cuda(async=self.config.async_loading)
            corner = data['corner'].float().cuda(async=self.config.async_loading)
            box = data['box'].float().cuda(async=self.config.async_loading)

            reg_out = self.reg(torch.cat((edge, corner), dim=1))

            loss = self.mse(reg_out, box)
            loss.backward()

            self.opt.step()
                
            avg_loss.update(loss)

        tqdm_batch.close()

        self.summary_writer.add_scalar('reg/loss', avg_loss.val, self.epoch)

        self.scheduler.step(avg_loss.val)

        self.logger.warning('info - lr: {}, loss: {}'.format(get_lr(self.opt), avg_loss.val))

    def validate_by_epoch(self):
        tqdm_batch_val = tqdm(self.val_loader, total=self.val_iter, desc='epoch_val-{}'.format(self.epoch))

        with torch.no_grad():
            self.reg.eval()
            val_loss = AverageMeter()

            for curr_it, data in enumerate(tqdm_batch_val):
                edge = data['edge'].float().cuda(async=self.config.async_loading)
                corner = data['corner'].float().cuda(async=self.config.async_loading)
                box = data['box'].float().cuda(async=self.config.async_loading)

                reg_out = self.reg(torch.cat((edge, corner), dim=1))

                loss = self.mse(reg_out, box)

                val_loss.update(loss)

            tqdm_batch_val.close()

            self.summary_writer.add_scalar('reg_val/loss', val_loss.val, self.epoch)

            if val_loss.val < self.best_val_loss:
                self.best_val_loss = val_loss.val
                if self.epoch > self.pretraining_step_size:
                    self.save_checkpoint()
