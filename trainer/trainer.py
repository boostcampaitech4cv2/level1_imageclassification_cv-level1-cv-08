import numpy as np
import torch
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score
from torchvision.utils import make_grid
from tqdm import tqdm
import wandb
from base import BaseTrainer
from utils import inf_loop


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

        self.train_acc = MulticlassAccuracy(num_classes=18).to(device)
        self.valid_acc = MulticlassAccuracy(num_classes=18).to(device)
        self.train_f1 = MulticlassF1Score(num_classes=18,average='macro').to(device)
        self.valid_f1 = MulticlassF1Score(num_classes=18,average='macro').to(device)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        progress = tqdm(
            self.data_loader,
            total=len(self.data_loader),
            desc=f"Epoch {epoch:2d} {'Train...':<12} ",
            ncols="80%",
            dynamic_ncols=True,
            ascii=True,
        )
        train_total_loss = 0
        for batch_idx, (data, target) in enumerate(progress):
            data, label = data.to(self.device), target[0].to(self.device)

            self.optimizer.zero_grad()
            output = self.model(data)
            pred = torch.argmax(output, dim=1)

            acc = self.train_acc(pred, label)
            f1 = self.train_f1(pred, label)

            loss = self.criterion(output, label)
            loss.backward()
            self.optimizer.step()

            train_total_loss += loss.item()
            progress.set_postfix(Loss=loss.item(), Acc=acc.item(), F1=f1.item())
            if batch_idx == self.len_epoch:
                break
        train_total_loss /= len(self.data_loader)
        wandb.log(
            {
                "train/acc": (train_total_acc := self.train_acc.compute()),
                "train/f1": (train_total_f1 := self.train_f1.compute()),
                "train/loss": train_total_loss,
                "Epoch": epoch,
                "Learning Rate": self.optimizer.param_groups[0]["lr"],
            }
        )
        self.logger.info(
            f"Epoch {epoch:2d} Train... Loss: {train_total_loss:.4f}// Acc: {train_total_acc:.4f}// F1: {train_total_f1:.4f}"
        )
        self.train_acc.reset()
        self.train_f1.reset()

        if self.do_validation:
            valid_total_loss = self._valid_epoch(epoch)

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return {"train/loss": train_total_loss, "val/loss": valid_total_loss}

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        progress = tqdm(
            self.valid_data_loader,
            total=len(self.valid_data_loader),
            desc=f"Epoch {epoch:2d} {'Valid...':<12} ",
            ncols="80%",
            dynamic_ncols=True,
            ascii=True,
        )
        valid_total_loss = 0
        with torch.no_grad():
            for data, target in progress:
                # target[0]: label, target[1]: mask, target[2]: gender, target[3]: age
                data, label = data.to(self.device), target[0].to(self.device)

                output = self.model(data)
                pred = torch.argmax(output, dim=1)

                acc = self.valid_acc(pred, label)
                f1 = self.valid_f1(pred, label)

                loss = self.criterion(output, label)
                valid_total_loss += loss.item()
                progress.set_postfix(Loss=loss.item(), Acc=acc.item(), F1=f1.item())
        valid_total_loss /= len(self.valid_data_loader)

        wandb.log(
            {
                "valid/acc": (valid_total_acc := self.valid_acc.compute()),
                "valid/f1": (valid_total_f1 := self.valid_f1.compute()),
                "valid/loss": valid_total_loss,
            }
        )
        self.valid_acc.reset()
        self.valid_f1.reset()
        self.logger.info(
            f"Epoch {epoch:2d} Valid... Loss: {valid_total_loss:.4f}// Acc: {valid_total_acc:.4f}// F1: {valid_total_f1:.4f}"
        )

        return valid_total_loss
