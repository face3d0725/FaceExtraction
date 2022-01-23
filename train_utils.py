import torch
from tqdm import tqdm
from meter import AverageValueMeter
from Dataset.utils import tensor2img
import sys
import numpy as np
import pathlib
import os
import cv2


class Epoch:
    def __init__(self, model, loss, metrics, stage_name, device='cpu', sv_pth=None, show_step=500, verbose=True):
        self.model = model
        self.loss = loss
        self.metrics = metrics
        self.stage_name = stage_name
        self.verbose = verbose
        self.device = device
        self.it = 0
        self.show_step = show_step
        if not sv_pth:
            pth = pathlib.Path(__file__).parent.absolute()
            sv_folder = 'res'
            sv_root = os.path.join(pth, sv_folder)
            self.sv_root = sv_root
            if not os.path.exists(sv_root):
                os.mkdir(sv_root)

    def _format_logs(self, logs):
        str_logs = ['{} - {:.4}'.format(k, v) for k, v in logs.items()]
        s = ', '.join(str_logs)
        return s

    def batch_update(self, x, y):
        raise NotImplementedError

    def on_epoch_start(self):
        pass

    def run(self, dataloader):
        self.on_epoch_start()
        logs = {}
        loss_meter = AverageValueMeter()
        metric_meters = {metric.__name__: AverageValueMeter() for metric in self.metrics}
        with tqdm(dataloader, desc=self.stage_name, file=sys.stdout, disable=not self.verbose) as iterator:
            for x, y in iterator:
                self.it += 1
                x, y = x.to(self.device), y.to(self.device)
                loss, y_pred = self.batch_update(x, y)
                loss_value = loss.item()
                loss_meter.add(loss_value)
                loss_logs = {self.loss.__name__: loss_meter.mean}
                logs.update(loss_logs)

                for metric_fn in self.metrics:
                    metric_value = metric_fn(y_pred, y).item()
                    metric_meters[metric_fn.__name__].add(metric_value)

                metric_logs = {k: v.mean for k, v in metric_meters.items()}
                logs.update(metric_logs)

                if self.verbose:
                    s = self._format_logs(logs)
                    iterator.set_postfix_str(s)

                if self.it % self.show_step == 0:
                    with torch.no_grad():
                        pred_mask = (y_pred > 0).type(torch.float32)
                        face_pred = tensor2img(x * pred_mask)
                        face_gt = tensor2img(x * y)
                        img = tensor2img(x)
                        show = np.concatenate((img, face_gt, face_pred), axis=0)
                        sv_name = f'{self.stage_name}_it_{self.it}.png'
                        sv_pth = os.path.join(self.sv_root, sv_name)
                        cv2.imwrite(sv_pth, show[:,:,::-1])

        return logs


class TrainEpoch(Epoch):
    def __init__(self, model, loss, metrics, optimizer, device='cpu', verbose=True):
        super().__init__(model=model,
                         loss=loss,
                         metrics=metrics,
                         stage_name='train',
                         device=device,
                         verbose=verbose)

        self.optimizer = optimizer

    def on_epoch_start(self):
        self.model.train()

    def batch_update(self, x, y):
        self.optimizer.zero_grad()
        prediction = self.model(x)
        loss = self.loss(prediction, y)
        loss.backward()
        self.optimizer.step()
        return loss, prediction


class ValidEpoch(Epoch):
    def __init__(self, model, loss, metrics, device='cpu', verbose=True):
        super().__init__(
            model=model,
            loss=loss,
            metrics=metrics,
            stage_name='valid',
            device=device,
            verbose=verbose,
            show_step=200
        )

    def on_epoch_start(self):
        self.model.eval()

    def batch_update(self, x, y):

        with torch.no_grad():
            prediction = self.model(x)
            loss = self.loss(prediction, y)

        return loss, prediction


if __name__ == '__main__':
    epoch = Epoch(None, None, None, None)
