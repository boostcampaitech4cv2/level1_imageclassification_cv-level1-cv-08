import numpy as np
import torch
from torchmetrics import Accuracy, F1Score
from torchvision.utils import make_grid
from tqdm import tqdm
import wandb
from base import BaseTrainer
from utils import inf_loop, MetricTracker


class Trainer(BaseTrainer):
    """
    Trainer class
    """

    def __init__(
        self,
        model,
        criterion,
        optimizer,
        config,
        device,
        data_loader,
        valid_data_loader=None,
        lr_scheduler=None,
        len_epoch=None,
    ):
        super().__init__(model, criterion, optimizer, config)
        self.config = config
        self.device = device
        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size)) * 2

        self.train_metrics = MetricTracker(
            "train/loss",
            writer=self.writer,
        )

        self.valid_metrics = MetricTracker(
            "val/loss",
            writer=self.writer,
        )
        self.train_acc = Accuracy().to(device)
        self.valid_acc = Accuracy().to(device)
        self.train_f1 = F1Score(avergage="macro", num_classes=18).to(device)
        self.valid_f1 = F1Score(average="macro", num_classes=18).to(device)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        progress = tqdm(
            self.data_loader,
            total=len(self.data_loader),
            desc=f"Epoch {epoch:2d} {'Train...':<12} ",
            ncols=200,
        )
        for batch_idx, (data, target) in enumerate(progress):
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(data)
            pred = torch.argmax(output, dim=1)

            acc = self.train_acc(pred, target)
            f1 = self.train_f1(pred, target)

            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()

            self.train_metrics.update("train/loss", loss.item())
            progress.set_postfix(Loss=loss.item(), Acc=acc.item(), F1=f1.item())
            if batch_idx == self.len_epoch:
                break
        log = self.train_metrics.result()
        wandb.log(
            {
                "train/acc": self.train_acc.compute(),
                "train/f1": self.train_f1.compute(),
                "train/loss": log["train/loss"],
                "Epoch": epoch,
                "Learning Rate": self.optimizer.param_groups[0]["lr"],
            }
        )
        self.train_acc.reset()
        self.train_f1.reset()

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{f"{k}": v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()
        progress = tqdm(
            self.valid_data_loader,
            total=len(self.valid_data_loader),
            desc=f"Epoch {epoch:2d} {'Valid...':<12} ",
            ncols=200,
        )
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(progress):
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                pred = torch.argmax(output, dim=1)

                acc = self.valid_acc(pred, target)
                f1 = self.valid_f1(pred, target)

                loss = self.criterion(output, target)
                self.valid_metrics.update("val/loss", loss.item())
                progress.set_postfix(Loss=loss.item(), Acc=acc.item(), F1=f1.item())

        wandb.log(
            {
                "valid/acc": self.valid_acc.compute(),
                "valid/f1": self.valid_f1.compute(),
                "valid/loss": self.valid_metrics.result()["val/loss"],
            }
        )
        self.valid_acc.reset()
        self.valid_f1.reset()

        return self.valid_metrics.result()

    def _progress(self, batch_idx, train=True):
        if train:
            if hasattr(self.data_loader, "n_samples"):
                current = batch_idx * self.data_loader.batch_size
                total = self.data_loader.n_samples
            else:
                current = batch_idx + 1
                total = self.len_epoch

            return f"[{current}/{total} ({100.0 * current / total:.0f}%)]"

        if hasattr(self.valid_data_loader, "n_samples"):
            current = batch_idx * self.valid_data_loader.batch_size
            total = self.valid_data_loader.n_samples
        else:
            current = batch_idx + 1
            total = len(self.valid_data_loader)

        return f"[{current}/{total} ({100.0 * current / total:.0f}%)]"
