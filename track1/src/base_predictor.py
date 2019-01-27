import torch
import torch.utils.data.dataloader
from torch.utils.data.dataloader import default_collate
from tqdm import tqdm


class BasePredictor():
    def __init__(self,
                 batch_size=10,
                 max_epochs=10,
                 valid=None,
                 device=None,
                 metrics={},
                 learning_rate=1e-3,
                 max_iters_in_epoch=1e20):
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.valid = valid
        self.metrics = metrics
        self.learning_rate = learning_rate
        self.max_iters_in_epoch = max_iters_in_epoch

        if device is not None:
            self.device = torch.device(device)
        else:
            self.device = torch.device('cuda:0' if torch.cuda.is_available()
                                       else 'cpu')

        self.epoch = 0

    def fit_dataset(self, data, collate_fn=default_collate, callbacks=[]):
        # Start the training loop.
        while self.epoch < self.max_epochs:

            # train and evaluate train score
            print('training %i' % self.epoch)
            dataloader = torch.utils.data.DataLoader(
                data,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=1,
                collate_fn=collate_fn)
            # train epoch
            log_train = self._run_epoch(dataloader, True)

            # evaluate valid score
            if self.valid is not None:
                print('evaluating %i' % self.epoch)
                dataloader = torch.utils.data.DataLoader(
                    self.valid,
                    batch_size=self.batch_size,
                    shuffle=False,
                    collate_fn=collate_fn,
                    num_workers=1)
                # evaluate model
                log_valid = self._run_epoch(dataloader, False)
            else:
                log_valid = None

            for callback in callbacks:
                callback.on_epoch_end(log_train, log_valid, self)

            self.epoch += 1

    def predict_dataset(self, data,
                        collate_fn=default_collate,
                        batch_size=None,
                        predict_fn=None):
        if batch_size is None:
            batch_size = self.batch_size
        if predict_fn is None:
            predict_fn = self._predict_batch

        # set model to eval mode
        self.model.eval()

        # make dataloader
        dataloader = torch.utils.data.DataLoader(
            data,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=1)

        ys_ = []
        with torch.no_grad():
            for batch in tqdm(dataloader):
                batch_y_ = predict_fn(batch)
                ys_.append(batch_y_)

        ys_ = torch.cat(ys_, 0)

        return ys_

    def save(self, path):
        torch.save({
            'epoch': self.epoch + 1,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }, path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.epoch = checkpoint['epoch']
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

    def _run_epoch(self, dataloader, training):
        # set model training/evaluation mode
        self.model.train(training)

        # run batches for train
        loss = 0

        # reset metric accumulators
        for metric in self.metrics:
            metric.reset()

        if training:
            iter_in_epoch = min(len(dataloader), self.max_iters_in_epoch)
            description = 'training'
        else:
            iter_in_epoch = len(dataloader)
            description = 'evaluating'

        # run batches
        trange = tqdm(enumerate(dataloader),
                      total=iter_in_epoch,
                      desc=description)
        for i, batch in trange:
            if training and i >= iter_in_epoch:
                break

            if training:
                output, batch_loss = \
                    self._run_iter(batch, training)
                self.optimizer.zero_grad()
                batch_loss.backward()
                self.optimizer.step()
            else:
                with torch.no_grad():
                    output, batch_loss = \
                        self._run_iter(batch, training)
                    self.optimizer.zero_grad()

            # accumulate loss and metric scores
            loss += batch_loss.item()
            for metric in self.metrics:
                metric.update(output.data, batch)
            trange.set_postfix(loss=loss / (i + 1))

        # calculate averate loss and metrics
        loss /= iter_in_epoch

        epoch_log = {}
        epoch_log['loss'] = float(loss)
        for metric in self.metrics:
            score = metric.get_score()
            print('{}: {} '.format(metric.name, score))
            epoch_log[metric.name] = score
        print('loss=%f\n' % loss)
        return epoch_log

    def _run_iter(self, batch, training):
        """ Run iteration for training.

        Args:
            batch (dict)
            training (bool)

        Returns:
            predicts: Prediction of the batch.
            loss (FloatTensor): Loss of the batch.
        """
        pass

    def _predict_batch(self, batch):
        """ Run iteration for predicting.

        Args:
            batch (dict)

        Returns:
            predicts: Prediction of the batch.
        """
        pass
