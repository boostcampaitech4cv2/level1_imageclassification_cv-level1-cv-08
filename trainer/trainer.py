import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import make_grid
from tqdm import tqdm
import wandb
from base import BaseTrainer
from utils import inf_loop, MetricTracker, get_age, get_gender, get_mask
from TRACER.model.TRACER import TRACER
from TRACER.util.utils import load_pretrained
from TRACER.config import getConfig
import cv2
from torchvision import transforms
import os
from datetime import datetime


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
            *["loss", "mask_loss", "gender_loss", "age_loss"],
            *[m.__name__ for m in self.metric_ftns],
            writer=self.writer,
        )
        self.valid_metrics = MetricTracker(
            *["loss", "mask_loss", "gender_loss", "age_loss"],
            *[m.__name__ for m in self.metric_ftns],
            writer=self.writer,
        )
        self.args = getConfig()
        self.tracer = nn.DataParallel(TRACER(self.args)).to(device)
        path = load_pretrained(f"TE-{self.args.arch}")
        self.tracer.load_state_dict(path)

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
        self.train_metrics.reset()
        for batch_idx, (data, original_size, target) in enumerate(progress):
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
            faces = self.setup_face(data, original_size)

            outputs = self.model(faces)
            # output[0]: mask, output[1]: gender, output[2]: age
            pred = (
                get_mask(outputs) + get_gender(outputs) + get_age(outputs, self.device)
            )

            # if you don't need inner_weight, please set null in config.json.
            loss_list = [
                self.criterion(name, output, feature, inner_weight, loss_weight)
                for (name, inner_weight, loss_weight), output, feature in zip(
                    self.config["loss_name"], outputs, feature_labels
                )
            ]
            mask_loss = loss_list[0]
            gender_loss = loss_list[1]
            age_loss = loss_list[2]

            loss = sum(loss_list)
            loss.backward()
            self.optimizer.step()
            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update("loss", loss.item())
            self.train_metrics.update("mask_loss", mask_loss.item())
            self.train_metrics.update("gender_loss", gender_loss.item())
            self.train_metrics.update("age_loss", age_loss.item())
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
                    "train/total_loss": batch_log["loss"],
                    "train/mask_loss": batch_log["mask_loss"],
                    "train/gender_loss": batch_log["gender_loss"],
                    "train/age_loss": batch_log["age_loss"],
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
            for (data, original_size, target) in progress:
                # target[0]: label, target[1]: mask, target[2]: gender, target[3]: age

                data, target, mask, gender, age = (
                    data.to(self.device),
                    target[0].to(self.device),
                    target[1].to(self.device),
                    target[2].to(self.device),
                    target[3].to(self.device),
                )
                feature_labels = [mask, gender, age]
                outputs = self.model(data)
                # output[0]: mask, output[1]: gender, output[2]: age
                pred = (
                    get_mask(outputs)
                    + get_gender(outputs)
                    + get_age(outputs, self.device)
                )

                loss_list = [
                    self.criterion(name, output, feature, inner_weight, loss_weight)
                    for (name, inner_weight, loss_weight), output, feature in zip(
                        self.config["loss_name"], outputs, feature_labels
                    )
                ]
                mask_loss = loss_list[0]
                gender_loss = loss_list[1]
                age_loss = loss_list[2]

                loss = sum(loss_list)
                self.valid_metrics.update("loss", loss.item())
                self.train_metrics.update("mask_loss", mask_loss.item())
                self.train_metrics.update("gender_loss", gender_loss.item())
                self.train_metrics.update("age_loss", age_loss.item())
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
                    "val/total_loss": batch_log["loss"],
                    "val/mask_loss": batch_log["mask_loss"],
                    "val/gender_loss": batch_log["gender_loss"],
                    "val/age_loss": batch_log["age_loss"],
                }
            )
        self.logger.info(
            f"Epoch {epoch:2d} Valid... Loss: {batch_log['loss']:4f}// Acc: {batch_log['accuracy']:4f}// F1: {batch_log['f1']:4f}"
        )

        return batch_log

    def denormalize(self, image):
        inv_norm = transforms.Compose(
            [
                transforms.Normalize(
                    mean=[0.0, 0.0, 0.0], std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
                ),
                transforms.Normalize(
                    mean=[-0.485, -0.456, -0.406], std=[1.0, 1.0, 1.0]
                ),
            ]
        )
        return inv_norm(image)

    def setup_face(self, data, original_size):
        no_bg, _, _ = self.tracer(data)
        H, W = original_size
        faces = self.make_face(H, W, no_bg, data)
        faces = faces.to(self.device)
        return faces

    def masking(self, original_image, masking, height, width):
        original_image = self.denormalize(original_image)

        original_image = F.interpolate(
            original_image.unsqueeze(0), size=(height, width), mode="bilinear"
        )
        original_image = (
            original_image.squeeze().permute(1, 2, 0).detach().cpu().numpy() * 255.0
        ).astype(np.uint8)
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        self.original_image = original_image

        masking = cv2.bitwise_and(original_image, original_image, mask=masking)
        return masking

    def make_face(self, H, W, no_bg, data):
        faces = []
        for i in range(data.size(0)):
            h, w = H[i].item(), W[i].item()
            output = F.interpolate(no_bg[i].unsqueeze(0), size=(h, w), mode="bilinear")

            output = (output.squeeze().detach().cpu().numpy() * 255.0).astype(np.uint8)
            salient_objects = self.masking(data[i], output, h, w)
            face = self.filter_face(salient_objects)
            face = self.cut_face(face)

            faces.append(face)
        return torch.stack(faces)

    def filter_face(self, img):
        imgray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # 스레시홀드로 바이너리 이미지로 만들어서 검은배경에 흰색전경으로 반전
        _, imthres = cv2.threshold(imgray, 0, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(
            imthres, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if len(contours) > 1:
            contour_sizes = [(cv2.contourArea(ctr), ctr) for ctr in contours]
            max_contour = max(contour_sizes, key=lambda x: x[0])[1]
        elif len(contours) == 1:
            max_contour = contours[0]
        else:
            return self.original_image

        # 결과 출력
        tmp_img = np.zeros_like(img)
        cv2.drawContours(tmp_img, [max_contour], 0, (255, 255, 255), cv2.FILLED)
        tmp_img = cv2.bitwise_and(img, tmp_img)
        return tmp_img

    def cut_face(self, face):
        face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        positions = np.nonzero(face_gray)
        top, bottom = positions[0].min(), positions[0].max()
        left, right = positions[1].min(), positions[1].max()
        face = face[top:bottom, left:right]
        # cv2.imwrite(
        #     os.path.join(os.getcwd(), "object", f"{str(datetime.now())}.png"),
        #     face,
        # )

        face = self.data_loader.dataset.transform(image=face)["image"]
        return face

        # salient_object = self.post_processing(data[i], output, h, w)
        # cv2.imwrite(
        #     os.path.join(os.getcwd(), "mask", f"{target[i]}.png"),
        #     output,
        # )
        # cv2.imwrite(
        #     os.path.join(os.getcwd(), "object", f"{target[i]}.png"),
        #     salient_object,
        # )
