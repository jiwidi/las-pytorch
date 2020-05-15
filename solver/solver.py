import os
import time
from torch.utils.tensorboard import SummaryWriter
import torch


class Solver(object):
    """
    """

    def __init__(self, data, model, optimizer, params):
        self.tr_loader = data['tr_loader']
        self.cv_loader = data['cv_loader']
        self.model = model
        self.optimizer = optimizer
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Training config
        self.epochs = params['epochs']
        self.half_lr = params['half_lr']
        self.early_stop = params['early_stop']
        self.max_norm = params['max_norm']
        # # save and load model
        self.save_folder = params['save_folder']
        self.checkpoint = params['checkpoint']
        self.continue_from = params['continue_from']
        # self.model_path = args.model_path
        # logging
        self.print_freq = params['print_freq']
        self.tensorboard = params['tensorboard']
        self.writer = SummaryWriter()
        #
        self.tr_loss = torch.Tensor(self.epochs)
        self.cv_loss = torch.Tensor(self.epochs)
        # self.visdom = args.visdom
        # self.visdom_id = args.visdom_id
        # if self.visdom:
        #     from visdom import Visdom
        #     self.vis = Visdom(env=self.visdom_id)
        #     self.vis_opts = dict(title=self.visdom_id,
        #                          ylabel='Loss', xlabel='Epoch',
        #                          legend=['train loss', 'cv loss'])
        #     self.vis_window = None
        #     self.vis_epochs = torch.arange(1, self.epochs + 1)

        self._reset()

    def _reset(self):
        # Reset
        if self.continue_from:
            print('Loading checkpoint model %s' % self.continue_from)
            package = torch.load(self.continue_from)
            self.model.load_state_dict(package['state_dict'])
            self.optimizer.load_state_dict(package['optim_dict'])
            self.start_epoch = int(package.get('epoch', 1))
            try:
                self.tr_loss[:self.start_epoch] = package['tr_loss'][:self.start_epoch]
                self.cv_loss[:self.start_epoch] = package['cv_loss'][:self.start_epoch]
            except:
                pass
        else:
            self.start_epoch = 0
        # Create save folder
        os.makedirs(self.save_folder, exist_ok=True)
        self.prev_val_loss = float("inf")
        self.best_val_loss = float("inf")
        self.halving = False

    def train(self):
        # Train model multi-epoches
        for epoch in range(self.start_epoch, self.epochs):
            # Train one epoch
            print("Training...")
            self.model.train()  # Turn on BatchNorm & Dropout
            start = time.time()
            tr_avg_loss = self._run_one_epoch(epoch)
            print('-' * 85)
            print('Train Summary | End of Epoch {0} | Time {1:.2f}s | '
                  'Train Loss {2:.3f}'.format(
                      epoch + 1, time.time() - start, tr_avg_loss))
            print('-' * 85)

            # Save model each epoch
            if self.checkpoint:
                file_path = os.path.join(
                    self.save_folder, 'epoch%d.pth.tar' % (epoch + 1))
                torch.save(self.model.serialize(self.model, self.optimizer, epoch + 1,
                                                tr_loss=self.tr_loss[epoch],
                                                cv_loss=self.cv_loss[epoch]),
                           file_path)
                print('Saving checkpoint model to %s' % file_path)

            # Cross validation
            print('Cross validation...')
            self.model.eval()  # Turn off Batchnorm & Dropout
            val_loss = self._run_one_epoch(epoch, cross_valid=True)
            print('-' * 85)
            print('Valid Summary | End of Epoch {0} | Time {1:.2f}s | '
                  'Valid Loss {2:.3f}'.format(
                      epoch + 1, time.time() - start, val_loss))
            print('-' * 85)

            # Adjust learning rate (halving)
            if self.half_lr and val_loss >= self.prev_val_loss:
                if self.early_stop and self.halving:
                    print("Already start halving learing rate, it still gets "
                          "too small imporvement, stop training early.")
                    break
                self.halving = True
            if self.halving:
                optim_state = self.optimizer.state_dict()
                optim_state['param_groups'][0]['lr'] = \
                    optim_state['param_groups'][0]['lr'] / 2.0
                self.optimizer.load_state_dict(optim_state)
                print('Learning rate adjusted to: {lr:.6f}'.format(
                    lr=optim_state['param_groups'][0]['lr']))
            self.prev_val_loss = val_loss

            # Save the best model
            self.tr_loss[epoch] = tr_avg_loss
            self.cv_loss[epoch] = val_loss
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                file_path = os.path.join(
                    self.save_folder, 'BESTMODEL-epoch%d.pth.tar' % (epoch + 1))
                torch.save(self.model.serialize(self.model, self.optimizer, epoch + 1,
                                                tr_loss=self.tr_loss,
                                                cv_loss=self.cv_loss),
                           file_path)
                print("Find better validated model, saving to %s" % file_path)

    def _run_one_epoch(self, epoch, cross_valid=False):
        start = time.time()
        total_loss = 0

        data_loader = self.tr_loader if not cross_valid else self.cv_loader

        for i, (data) in enumerate(data_loader):
            utt_ids, feature, label = data
            sample_id = data[0]
            t_inputs = data[1]
            t_targets = data[2]
            padded_input = t_inputs['inputs'].to(self.device)
            inputs_length = t_inputs['inputs_length'].to(self.device)
            padded_target = t_targets['targets'].to(self.device)
            target_length = t_targets['targets_length'].to(self.device)

            loss = self.model(padded_input, inputs_length, padded_target)

            if not cross_valid:
                self.optimizer.zero_grad()
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                                           self.max_norm)
                self.optimizer.step()

            total_loss += loss.item()

            if i % self.print_freq == 0:
                print('Epoch {0} | Iter {1} | Average Loss {2:.3f} | '
                      'Current Loss {3:.6f} | {4:.1f} ms/batch'.format(
                          epoch + 1, i + 1, total_loss / (i + 1),
                          loss.item(), 1000 * (time.time() - start) / (i + 1)),
                      flush=True)

            #Visualizing iteration loss using tensorboard
            if (self.tensorboard):
                mode = 'test' if cross_valid else 'train'
                self.writer.add_scalar(f'Iteration-Loss/{mode}', loss, i)
        #Visualizing epoch loss using tensorboard
        if (self.tensorboard):
                mode = 'test' if cross_valid else 'train'
                self.writer.add_scalar(f'Epoch-Loss/{mode}', total_loss / (i + 1), epoch)
        return total_loss / (i + 1)