import numpy as np
import torch
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
        metric_ftns,
        optimizer,
        config,
        device,
        data_loader,
        valid_data_loader=None,
        lr_scheduler=None,
        len_epoch=None,
    ):
        super().__init__(model, criterion, metric_ftns, optimizer, config)
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
        self.log_step = int(np.sqrt(data_loader.batch_size))

        self.train_metrics = MetricTracker(
            "loss", *[m.__name__ for m in self.metric_ftns], writer=self.writer
        )
        self.valid_metrics = MetricTracker(
            "loss", *[m.__name__ for m in self.metric_ftns], writer=self.writer
        )

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch
        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        if self.model.layer_list:
            self.model.train_layer(epoch)
            trainable_params = filter(
                lambda p: p.requires_grad, self.model.parameters()
            )
            self.optimizer = self.config.init_obj(
                "optimizer", torch.optim, trainable_params
            )
        self.model.train()
        progress = tqdm(
            self.data_loader,
            total=len(self.data_loader),
            desc=f"Epoch {epoch:2d} {'Train...':<12} ",
            ncols="80%",
            dynamic_ncols=True,
            ascii=True,
        )
        self.train_metrics.reset()
        for batch_idx, (data, target) in enumerate(progress):
            # target[0]: label, target[1]: mask, target[2]: gender, target[3]: age
            data, target, mask, gender, age = (
                data.to(self.device),
                target[0].to(self.device),
                target[1].to(self.device),
                target[2].to(self.device),
                target[3].to(self.device),
            )
            feature_labels = [mask, gender, age]
            self.optimizer.zero_grad()
            output = self.model(data)
            outputs = [output[0], output[1], output[2]]
            # output[0]: mask, output[1]: gender, output[2]: age
            pred = (
                self.get_mask(output) + self.get_gender(output) + self.get_age(output)
            )

            # if you don't need inner_weight, please set null in config.json.
            loss_list = [
                self.criterion(name, output, feature, inner_weight, loss_weight)
                for (name, inner_weight, loss_weight), output, feature in zip(
                    self.config["loss_name"], outputs, feature_labels
                )
            ]
            loss = sum(loss_list)
            loss.backward()
            self.optimizer.step()
            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update("loss", loss.item())
            for met in self.metric_ftns:
                self.train_metrics.update(met.__name__, (v := met(pred, target)))
                if met.__name__ == "f1":
                    f1 = v
                elif met.__name__ == "accuracy":
                    acc = v
            if batch_idx % self.log_step == 0 and self.config["visualize"]:
                self.writer.add_image(
                    "input", make_grid(data.cpu(), nrow=8, normalize=True)
                )

            progress.set_postfix(Loss=loss.item(), Acc=acc, F1=f1)
            if batch_idx == self.len_epoch:
                break
        batch_log = self.train_metrics.result()
        if self.config["wandb"]:
            wandb.log(
                {
                    "train/acc": batch_log["accuracy"],
                    "train/f1": batch_log["f1"],
                    "train/loss": batch_log["loss"],
                    "Epoch": epoch,
                    "Learning Rate": self.optimizer.param_groups[0]["lr"],
                }
            )
        self.logger.info(
            f"Epoch {epoch:2d} Train... Loss: {batch_log['loss']:4f}// Acc: {batch_log['accuracy']:4f}// F1: {batch_log['f1']:4f}"
        )
        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            batch_log.update(**{f"val/{k}": v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return batch_log

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
            ncols="80%",
            dynamic_ncols=True,
            ascii=True,
        )
        with torch.no_grad():
            for data, target in progress:
                # target[0]: label, target[1]: mask, target[2]: gender, target[3]: age

                data, target, mask, gender, age = (
                    data.to(self.device),
                    target[0].to(self.device),
                    target[1].to(self.device),
                    target[2].to(self.device),
                    target[3].to(self.device),
                )
                feature_labels = [mask, gender, age]
                output = self.model(data)
                outputs = [output[0], output[1], output[2]]
                # output[0]: mask, output[1]: gender, output[2]: age
                pred = (
                    self.get_mask(output)
                    + self.get_gender(output)
                    + self.get_age(output)
                )

                loss_list = [
                    self.criterion(name, output, feature, inner_weight, loss_weight)
                    for (name, inner_weight, loss_weight), output, feature in zip(
                        self.config["loss_name"], outputs, feature_labels
                    )
                ]
                loss = sum(loss_list)
                self.valid_metrics.update("loss", loss.item())
                for met in self.metric_ftns:
                    self.valid_metrics.update(met.__name__, (v := met(pred, target)))
                    if met.__name__ == "f1":
                        f1 = v
                    elif met.__name__ == "accuracy":
                        acc = v
                if self.config["visualize"]:
                    self.writer.add_image(
                        "input", make_grid(data.cpu(), nrow=8, normalize=True)
                    )

                progress.set_postfix(Loss=loss.item(), Acc=acc, F1=f1)
        batch_log = self.valid_metrics.result()
        if self.config["wandb"]:
            wandb.log(
                {
                    "val/acc": batch_log["accuracy"],
                    "val/f1": batch_log["f1"],
                    "val/loss": batch_log["loss"],
                }
            )
        self.logger.info(
            f"Epoch {epoch:2d} Valid... Loss: {batch_log['loss']:4f}// Acc: {batch_log['accuracy']:4f}// F1: {batch_log['f1']:4f}"
        )

        return batch_log

    def get_age(self, output):
        return torch.tensor(
            [0 if x <= 1 else (1 if x <= 4 else 2) for x in torch.argmax(output[2], -1)]
        ).to(self.device)

    def get_gender(self, output):
        return torch.argmax(output[1], -1) * 3

    def get_mask(self, output):
        return torch.argmax(output[0], -1) * 6
