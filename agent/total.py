import os
import shutil
import random
from tqdm import tqdm
from itertools import chain

import numpy as np

import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader

from tensorboardX import SummaryWriter

from graph.model.encoder import Encoder
from graph.model.decoder import Edge, Corner
from graph.model.regressor import Regressor
from graph.loss.loss import BCELoss, MSELoss
from data.dataset import Dataset

from utils.metrics import AverageMeter
from utils.train_utils import set_logger, count_model_prameters, get_lr


cudnn.benchmark = True


class Total(object):
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
        self.encoder = Encoder().cuda()
        self.edge = Edge().cuda()
        self.corner = Corner().cuda()
        self.reg = Regressor().cuda()

        # define loss
        self.bce = BCELoss().cuda()
        self.mse = MSELoss().cuda()

        # define lr
        self.lr = self.config.learning_rate

        # define optimizer
        self.opt = torch.optim.Adam([{'params': self.encoder.parameters()},
                                     {'params': self.edge.parameters()},
                                     {'params': self.corner.parameters()},
                                     {'params': self.reg.parameters()}],
                                    lr=self.lr, eps=1e-6)

        # define optimize scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.opt, mode='min', factor=0.8,
                                                                    cooldown=16, min_lr=8e-5)

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
        self.encoder = nn.DataParallel(self.encoder, device_ids=gpu_list)
        self.edge = nn.DataParallel(self.edge, device_ids=gpu_list)
        self.corner = nn.DataParallel(self.corner, device_ids=gpu_list)
        self.reg = nn.DataParallel(self.reg, device_ids=gpu_list)

        # Model Loading from the latest checkpoint if not found start from scratch.
        self.load_checkpoint(self.config.checkpoint_file)

        # Summary Writer
        self.summary_writer = SummaryWriter(log_dir=os.path.join(self.config.root_path, self.config.summary_dir),
                                            comment='LayoutNet')
        self.print_train_info()

    def print_train_info(self):
        print('seed: ', self.manual_seed)
        print('Number of model parameters: {}'.format(count_model_prameters(self.encoder)))
        print('Number of model parameters: {}'.format(count_model_prameters(self.reg)))

    def collate_function(self, samples):
        data = dict()

        data['img'] = torch.from_numpy(np.array([sample['img'] for sample in samples]))
        data['line'] = torch.from_numpy(np.array([sample['line'] for sample in samples]))
        data['corner'] = torch.from_numpy(np.array([sample['corner'] for sample in samples]))
        data['edge'] = torch.from_numpy(np.array([sample['edge'] for sample in samples]))
        data['box'] = torch.from_numpy(np.array([sample['box'] for sample in samples]))
        
        return data

    def load_checkpoint(self, file_name):
        filename = os.path.join(self.config.root_path, self.config.checkpoint_dir, file_name)
        try:
            print('Loading checkpoint {}'.format(filename))
            checkpoint = torch.load(filename)

            self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
            self.edge.load_state_dict(checkpoint['edge_state_dict'])
            self.corner.load_state_dict(checkpoint['corner_state_dict'])
            self.reg.load_state_dict(checkpoint['reg_state_dict'])

        except OSError as e:
            print('No checkpoint exists from {}. Skipping...'.format(self.config.checkpoint_dir))
            print('**First time to train**')

            filename = os.path.join(self.config.root_path, self.config.checkpoint_dir, 'corner_' + file_name)
            checkpoint = torch.load(filename)
            self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
            self.edge.load_state_dict(checkpoint['edge_state_dict'])
            self.corner.load_state_dict(checkpoint['corner_state_dict'])

            filename = os.path.join(self.config.root_path, self.config.checkpoint_dir, 'reg_' + file_name)
            checkpoint = torch.load(filename)
            self.reg.load_state_dict(checkpoint['reg_state_dict'])

    def save_checkpoint(self):
        tmp_name = os.path.join(self.config.root_path, self.config.checkpoint_dir,
                                'checkpoint_{}.pth.tar'.format(self.epoch))

        state = {
            'encoder_state_dict': self.encoder.state_dict(),
            'edge_state_dict': self.edge.state_dict(),
            'corner_state_dict': self.corner.state_dict(),
            'reg_state_dict': self.reg.state_dict(),
        }

        torch.save(state, tmp_name)
        shutil.copyfile(tmp_name, os.path.join(self.config.root_path, self.config.checkpoint_dir,
                                               self.config.checkpoint_file))

    def record_image(self, output, origin_corner, origin_edge, state='train'):
        self.summary_writer.add_image(state + '/origin_corner 1', origin_corner[0], self.epoch)
        self.summary_writer.add_image(state + '/origin_corner 2', origin_corner[1], self.epoch)

        self.summary_writer.add_image(state + '/origin_edge 1', origin_edge[0], self.epoch)
        self.summary_writer.add_image(state + '/origin_edge 2', origin_edge[1], self.epoch)

        self.summary_writer.add_image(state + '/model_corner 1', output[1][0], self.epoch)
        self.summary_writer.add_image(state + '/model_corner 2', output[1][1], self.epoch)

        self.summary_writer.add_image(state + '/model_edge 1', output[0][0], self.epoch)
        self.summary_writer.add_image(state + '/model_edge 2', output[0][1], self.epoch)

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
        corner, edge, out = None, None, None
        for curr_it, data in enumerate(tqdm_batch):
            self.encoder.train()
            self.edge.train()
            self.corner.train()
            self.reg.train()

            img = data['img'].float().cuda(async=self.config.async_loading)
            line = data['line'].float().cuda(async=self.config.async_loading)
            box = data['box'].float().cuda(async=self.config.async_loading)
            edge = data['edge'].float().cuda(async=self.config.async_loading)
            corner = data['corner'].float().cuda(async=self.config.async_loading)

            encoder_out_list = self.encoder(torch.cat((img, line), dim=1))
            edge_out_list, edge_out = self.edge(encoder_out_list)
            corner_out = self.corner(edge_out_list)

            reg_out = self.reg(torch.cat((edge_out, corner_out), dim=1))

            loss = self.bce(edge_out, edge)
            loss[edge > 0.] *= 4
            loss = loss.mean()

            c_loss = self.bce(corner_out, corner)
            c_loss[corner > 0.] *= 4
            loss += c_loss.mean()

            loss += self.mse(reg_out, box) * 0.01
            out = (edge_out, corner_out)

            self.opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(chain(self.encoder.parameters(), self.edge.parameters(),
                                           self.corner.parameters(), self.reg.parameters()),
                                     3.0, norm_type='inf')
            self.opt.step()

            avg_loss.update(loss)

        tqdm_batch.close()

        self.record_image(out, corner, edge)
        self.summary_writer.add_scalar('train/loss', avg_loss.val, self.epoch)

        self.scheduler.step(avg_loss.val)

        self.logger.warning('info - lr: {}, loss: {}'.format(get_lr(self.opt), avg_loss.val))

    def validate_by_epoch(self):
        val_loss = AverageMeter()
        corner, edge, out = None, None, None
        tqdm_batch_val = tqdm(self.val_loader, total=self.val_iter, desc='epoch_val-{}'.format(self.epoch))

        with torch.no_grad():
            self.encoder.eval()
            self.edge.eval()
            self.corner.eval()

            for curr_it, data in enumerate(tqdm_batch_val):
                img = data['img'].float().cuda(async=self.config.async_loading)
                line = data['line'].float().cuda(async=self.config.async_loading)
                edge = data['edge'].float().cuda(async=self.config.async_loading)
                corner = data['corner'].float().cuda(async=self.config.async_loading)

                encoder_out_list = self.encoder(torch.cat((img, line), dim=1))
                edge_out_list, edge_out = self.edge(encoder_out_list)
                corner_out = self.corner(edge_out_list)

                loss = self.bce(edge_out, edge)
                loss[edge > 0.] *= 4
                loss = loss.mean()

                c_loss = self.bce(corner_out, corner)
                c_loss[corner > 0.] *= 4
                loss += c_loss.mean()

                out = (edge_out, corner_out)

                val_loss.update(loss)

            tqdm_batch_val.close()

            self.record_image(out, corner, edge, 'val')
            self.summary_writer.add_scalar('val/loss', val_loss.val, self.epoch)

            if val_loss.val < self.best_val_loss:
                self.best_val_loss = val_loss.val
                if self.epoch > self.pretraining_step_size:
                    self.save_checkpoint()
