import argparse
import torch
from tqdm import tqdm
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
import pandas as pd


def main(config):
    logger = config.get_logger("test")

    # setup data_loader instances
    data_loader = getattr(module_data, config["data_loader"]["type"])(
        config["data_loader"]["args"]["data_dir"],
        batch_size=512,
        shuffle=False,
        validation_split=0.0,
        num_workers=2,
    )

    # build model architecture
    model = config.init_obj("arch", module_arch)
    # checkpoint = torch.load(config.resume)
    # state_dict = checkpoint["state_dict"]
    if config["n_gpu"] > 1:
        model = torch.nn.DataParallel(model)
    # model.load_state_dict(state_dict)

    # prepare model for testing
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    result = []
    with torch.no_grad():
        for data in tqdm(data_loader):
            data = data.to(device)
            mask_out, gender_out, age_out = model(data)
            output = (
                mask_out.data.max(1, keepdim=True)[1] * 6
                + gender_out.data.max(1, keepdim=True)[1] * 2
                + age_out.data.max(1, keepdim=True)[1]
            )
            result += list(map(lambda x: int(x), output))
    info_csv = pd.read_csv(config["data_loader"]["args"]["data_dir"] + "/info.csv")
    info_csv["ans"] = result
    info_csv.to_csv("/opt/ml/level1_imageclassification_cv-level1-cv-08/submision.csv")


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="PyTorch Template")
    args.add_argument(
        "-c",
        "--config",
        default=None,
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

    config = ConfigParser.from_args(args)
    main(config)
