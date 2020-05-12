import torch


class Runner(object):
    def __init__(self,data, model, optimizer, args):
        self.train_loader = data['train_loader']
        self.test_loader = data['test_loader']
        self.model = model
        self.optimizer = optimizer

        #Extra config for training
        self.epochs = args.epochs
        self.early_stop = args.early_stop

    def train(self)
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