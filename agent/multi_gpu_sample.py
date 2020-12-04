import os
import shutil
import random
from tqdm import tqdm

import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader

from tensorboardX import SummaryWriter

from graph.model.sample_model import SampleModel as Model
from graph.loss.sample_loss import SampleLoss as Loss
from data.sample_dataset import SampleDataset

from utils.metrics import AverageMeter
from utils.train_utils import free, frozen, set_logger, count_model_prameters


cudnn.benchmark = True


class Sample(object):
    def __init__(self, config):
        self.config = config
        self.flag_gan = False
        self.train_count = 0

        self.pretraining_step_size = self.config.pretraining_step_size
        self.batch_size = self.config.batch_size

        self.logger = set_logger('train_epoch.log')

        # define dataloader
        self.dataset = SampleDataset(self.config)
        self.dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False, num_workers=1,
                                     pin_memory=self.config.pin_memory, collate_fn=self.collate_function)

        # define models ( generator and discriminator)
        self.generator = Model().cuda()

        # define loss
        self.loss_generator = Loss().cuda()

        # define lr
        self.lr_generator = self.config.learning_rate

        # define optimizer
        self.opt_generator = torch.optim.Adam(self.generator.parameters(), lr=self.lr_generator)

        # define optimize scheduler
        self.scheduler_generator = torch.optim.lr_scheduler.ReduceLROnPlateau(self.opt_generator, mode='min',
                                                                              factor=0.8, cooldown=6)

        # initialize train counter
        self.epoch = 0
        self.accumulate_iter = 0
        self.total_iter = (len(self.dataset) + self.config.batch_size - 1) // self.config.batch_size

        self.manual_seed = random.randint(10000, 99999)

        torch.manual_seed(self.manual_seed)
        torch.cuda.manual_seed_all(self.manual_seed)
        random.seed(self.manual_seed)

        # parallel setting
        gpu_list = list(range(self.config.gpu_cnt))
        self.generator = nn.DataParallel(self.generator, device_ids=gpu_list)

        # Model Loading from the latest checkpoint if not found start from scratch.
        self.load_checkpoint(self.config.checkpoint_file)

        # Summary Writer
        self.summary_writer = SummaryWriter(log_dir=os.path.join(self.config.root_path, self.config.summary_dir),
                                            comment='BarGen')
        self.print_train_info()

    def print_train_info(self):
        print("seed: ", self.manual_seed)
        print('Number of model parameters: {}'.format(count_model_prameters(self.generator)))

    def collate_function(self, samples):
        return samples

    def load_checkpoint(self, file_name):
        filename = os.path.join(self.config.root_path, self.config.checkpoint_dir, file_name)
        try:
            print("Loading checkpoint '{}'".format(filename))
            checkpoint = torch.load(filename)

            self.generator.load_state_dict(checkpoint['generator_state_dict'])
            self.opt_generator.load_state_dict(checkpoint['generator_optimizer'])

        except OSError as e:
            print("No checkpoint exists from '{}'. Skipping...".format(self.config.checkpoint_dir))
            print("**First time to train**")

    def save_checkpoint(self, epoch):
        tmp_name = os.path.join(self.config.root_path, self.config.checkpoint_dir,
                                'checkpoint_{}.pth.tar'.format(epoch))

        state = {
            'generator_state_dict': self.generator.state_dict(),
            'generator_optimizer': self.opt_generator.state_dict(),
        }

        torch.save(state, tmp_name)
        shutil.copyfile(tmp_name, os.path.join(self.config.root_path, self.config.checkpoint_dir,
                                               self.config.checkpoint_file))

    def run(self):
        try:
            self.train()

        except KeyboardInterrupt:
            print("You have entered CTRL+C.. Wait to finalize")

    def train(self):
        for _ in range(self.config.epoch):
            self.epoch += 1
            self.train_by_epoch()

            if self.epoch > self.pretraining_step_size:
                self.save_checkpoint(self.config.checkpoint_file)

    def train_by_epoch(self):
        tqdm_batch = tqdm(self.dataloader, total=self.total_iter, desc="epoch-{}".format(self.epoch))

        avg_generator_loss = AverageMeter()
        for curr_it, (X, y) in enumerate(tqdm_batch):
            self.accumulate_iter += 1

            self.generator.train()
            free(self.generator)

            X = X.cuda(async=self.config.async_loading)

            logits = self.generator(X)

            loss = self.loss_disc(logits, y)
            loss.backward()
            self.opt_generator.step()
            avg_generator_loss.update(loss)

        tqdm_batch.close()

        self.scheduler_generator.step(avg_generator_loss.val)

        with torch.no_grad():
            self.generator.eval()

            # add evaluation code
