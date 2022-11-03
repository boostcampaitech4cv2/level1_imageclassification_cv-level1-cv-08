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
from utils import get_age, get_gender, get_mask
import ttach as tta
from data_loader.transform import train_transform, test_transform


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

    tta_transforms = tta.Compose(
        [
            tta.HorizontalFlip(),
            tta.Rotate90(angles=[0, 180]),
        ]
    )
    # t_transforms = train_transform(224)
    tta_model = tta.ClassificationTTAWrapper(model, tta_transforms).to(device)

    tta_model.eval()

    with torch.no_grad():
        for data in progress:
            data = data.to(device)
            output = tta_model(data)
            pred = get_mask(output) + get_gender(output) + get_age(output, device)
            preds.extend(pred.detach().cpu().numpy())

    # with torch.no_grad():
    #     for data in progress:
    #         data = data.to(device)  # image, label - age,gender,mask
    #         output = model(data)
    #         output = tta_model(data)

    #         # for transformer in tta_transforms:

    #         #     # augment image
    #         #     augmented_image = transformer.augment_image(data)

    #         #     # pass to model
    #         #     model_output = model(augmented_image, data)

    #         #     # reverse augmentation for mask and label
    #         #     deaug_mask = transformer.deaugment_mask(model_output['mask'])
    #         #     deaug_label = transformer.deaugment_label(model_output['label'])

    #         # output[0]: mask, output[1]: gender, output[2]: age
    #         pred = get_mask(output) + get_gender(output) + get_age(output, device)
    #         preds.extend(pred.detach().cpu().numpy())

    #         #
    #         # save sample images, or do something with output here
    #         #
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
        default="saved/models/Mask_base/11.03_01:47:43/best_epoch.pth",
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
