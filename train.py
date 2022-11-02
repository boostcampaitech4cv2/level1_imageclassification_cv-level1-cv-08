import argparse
import collections
import torch
import numpy as np
import albumentations as A
import wandb

import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
from trainer import Trainer
from utils import prepare_device
from pytorch_metric_learning import losses

# fix random seeds for reproducibility
SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


def main(CONFIG):
    logger = CONFIG.get_logger("train")

    # setup data_loader instances
    data_loader, valid_data_loader = CONFIG.init_ftn("data_loader", module_data)()

    # build model architecture, then print to console
    model = CONFIG.init_obj("arch", module_arch)
    logger.info(CONFIG["arch"]["args"]["model_name"])

    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(CONFIG["n_gpu"])
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    # get function handles of loss and metrics
    criterion = getattr(losses, CONFIG["loss"])()
    metrics = [getattr(module_metric, met) for met in CONFIG["metrics"]]

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = CONFIG.init_obj("optimizer", torch.optim, trainable_params)
    lr_scheduler = CONFIG.init_obj("lr_scheduler", torch.optim.lr_scheduler, optimizer)

    trainer = Trainer(
        model,
        criterion,
        metrics,
        optimizer,
        config=CONFIG,
        device=device,
        data_loader=data_loader,
        valid_data_loader=valid_data_loader,
        lr_scheduler=lr_scheduler,
    )
    if CONFIG["wandb"]:
        wandb.init(project="bcaitech4_mask_detection")
        wandb.config = {
            "lr": CONFIG["optimizer"]["args"]["lr"],
            "epochs": CONFIG["trainer"]["epochs"],
            "batch_size": CONFIG["data_loader"]["args"]["batch_size"],
        }
        wandb.run.name = (
            CONFIG["arch"]["args"]["model_name"]
            + "_"
            + str(CONFIG["data_loader"]["args"]["batch_size"])
            + "_"
            + str(CONFIG["lr_scheduler"]["type"])
        )
        wandb.run.save()
        # save data augmentation
        A.save(
            data_loader.dataset.transform,
            CONFIG.save_dir / "trn_trnsfrm.yml",
            data_format="yaml",
        )
        A.save(
            valid_data_loader.dataset.transform,
            CONFIG.save_dir / "vld_trnsfrm.yml",
            data_format="yaml",
        )
        wandb.save(str(CONFIG.save_dir / "*_trnsfrm.yml"))

    trainer.train()


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="PyTorch Template")
    args.add_argument(
        "-c",
        "--config",
        default="/opt/ml/level1_imageclassification_cv-level1-cv-08/config.json",
        type=str,
        help="config file path (default: None)",
    )
    args.add_argument(
        "-r",
        "--resume",
        default=None,
        type=str,
        help="path to latest checkpoint (default: None)",
    )
    args.add_argument(
        "-d",
        "--device",
        default=None,
        type=str,
        help="indices of GPUs to enable (default: all)",
    )

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple("CustomArgs", "flags type target")
    options = [
        CustomArgs(["--lr", "--learning_rate"], type=float, target="optimizer;args;lr"),
        CustomArgs(
            ["--bs", "--batch_size"], type=int, target="data_loader;args;batch_size"
        ),
        CustomArgs(
            ["--arch", "--architecture"], type=str, target="arch;args;model_name"
        ),
        CustomArgs(["--model", "--model_framework"], type=str, target="arch;type"),
    ]
    config = ConfigParser.from_args(args, options)
    main(config)
