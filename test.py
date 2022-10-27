import argparse
import collections
from datetime import datetime
import torch
import os
from tqdm import tqdm
import pandas as pd
import data_loader.data_loaders as module_data
import model.model as module_arch
from parse_config import ConfigParser


def main(CONFIG):
    logger = CONFIG.get_logger("test")

    # setup data_loader instances
    data_loader = getattr(module_data, CONFIG["data_loader"]["type"])(
        stage="eval",
        input_size=CONFIG["data_loader"]["args"]["input_size"],
        batch_size=CONFIG["data_loader"]["args"]["batch_size"],
        num_workers=CONFIG["data_loader"]["args"]["num_workers"],
    )

    # build model architecture
    model = CONFIG.init_obj("arch", module_arch)
    logger.info(CONFIG["arch"]["args"]["model_name"])

    logger.info(f"Loading checkpoint: {CONFIG.resume} ...")
    checkpoint = torch.load(CONFIG.resume)
    state_dict = checkpoint["state_dict"]
    if CONFIG["n_gpu"] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)
    logger.info("model loaded")

    # prepare model for testing
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    logger.info("Testing...")
    preds = []
    progress = tqdm(
        data_loader,
        total=len(data_loader),
        desc="Testing... ",
        ncols="80%",
        dynamic_ncols=True,
    )
    with torch.no_grad():
        for data in progress:
            data = data.to(device)
            output = model(data)
            pred = torch.argmax(output, dim=1)
            preds.extend(pred.detach().cpu().numpy())

            #
            # save sample images, or do something with output here
            #
    logger.info("Testing done.")

    info_path = os.path.join(CONFIG["info_dir"], "info.csv")
    submit = pd.read_csv(info_path)
    submit["ans"] = preds

    now = datetime.now().strftime(r"%m.%d_%H:%M:%S")
    submit_path = os.path.join(CONFIG["submit_dir"], f"{now}.csv")
    submit.to_csv(submit_path, index=False)

    logger.info(f"Submit saved to {submit_path}")


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="PyTorch Template")
    args.add_argument(
        "-c",
        "--config",
        default="config.json",
        type=str,
        help="config file path (default: None)",
    )
    args.add_argument(
        "-r",
        "--resume",
        default="/opt/ml/level1_imageclassification_cv-level1-cv-08/saved/models/Mask_base/10.27_01:26:26/model_best.pth",
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
        CustomArgs(
            ["--dtld", "--data_loader"], type=str, target="data_loader;args;stage"
        ),
    ]

    config = ConfigParser.from_args(args, options)
    main(config)
