import argparse
import torch
from tqdm import tqdm
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import cv2
import random
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
from PIL import Image
from torchvision import transforms
from torchvision.transforms import Resize, ToTensor, Normalize
from sklearn.manifold import TSNE
import numpy as np
import seaborn as sns

class TestDataset(Dataset):
    def __init__(self, img_paths, transform):
        self.img_paths = img_paths
        self.transform = transform

    def __getitem__(self, index):
        image = Image.open(self.img_paths[index])

        if self.transform:
            image = self.transform(image)
        return image

    def __len__(self):
        return len(self.img_paths)

def main(config, filename):
    logger = config.get_logger('submit')

    ## submit 용 ##
    test_dir = '/opt/ml/input/data/eval'
    submission = pd.read_csv(os.path.join(test_dir, 'info.csv'))
    image_dir = os.path.join(test_dir, 'images')

    image_paths = [os.path.join(image_dir, img_id) for img_id in submission.ImageID]
    transform = transforms.Compose([
        Resize((512, 384), Image.BILINEAR),
        ToTensor(),
        Normalize(mean=(0.5, 0.5, 0.5), std=(0.2, 0.2, 0.2)),
    ])
    dataset = TestDataset(image_paths, transform)

    loader = DataLoader(dataset, shuffle=False, batch_size=128, num_workers=2)
    ## submit 용 ##


    # build model architecture
    model = config.init_obj('arch', module_arch)
    logger.info(model)

    # get function handles of loss and metrics
    # loss_fn = getattr(module_loss, config['loss'])
    # metric_fns = [getattr(module_metric, met) for met in config['metrics']]

    logger.info('Loading checkpoint: {} ...'.format(config.resume))
    checkpoint = torch.load(config.resume)
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    all_predictions = []
    actual, deep_features = [], []

    for images in tqdm(loader):
        with torch.no_grad():
            images = images.to(device)
            outputs = model(images)
            pred = outputs.argmax(dim=-1)
            all_predictions.extend(pred.cpu().numpy())

            #for tSNE
            deep_features += outputs.cpu().tolist()
            _, preds = torch.max(outputs, 1)
            actual += preds.cpu().tolist()

    tsne = TSNE(n_components=2, random_state=0)
    cluster = np.array(tsne.fit_transform(np.array(deep_features)))
    actual = np.array(actual)

    # for i, label in enumerate(test_sets.classes):
    plt.figure(figsize=(16,16))
    for i, label in enumerate(range(3)):
        idx = np.where(actual==i)
        # print(type(idx), len(idx), idx)
        plt.scatter(cluster[idx,0], cluster[idx,1], marker='.', label=label)
    plt.title('label')
    plt.axis('off')
    plt.legend()
    plt.show()
    plt.savefig(f'visual/tSNE_org_{filename}.png')

    # plt.figure(figsize=(16,16))
    # age_list = [[], [], []]
    # for i, age in enumerate(range(100)):
    #     idx = np.where(actual==i)
    #     if age < 30:                age_list[0].extend(idx[0])
    #     if age >= 30 and age < 60:  age_list[1].extend(idx[0])
    #     if age >= 60:               age_list[2].extend(idx[0])

    # for i, idxs in enumerate(age_list):
    #     plt.scatter(cluster[idxs,0], cluster[idxs,1], marker='.', label=i)
    # plt.title('age')
    # plt.axis('off')
    # plt.legend()
    # plt.show()
    # plt.savefig(f'visual/tSNE_age_{filename}.png')


    submission['ans'] = all_predictions

    # 제출할 파일을 저장합니다.
    print('test inference is done!')
    print(submission['ans'].value_counts())

    plt.figure(figsize=(16,16))
    sns.displot(submission, x="ans", kind="kde", fill=True)
    plt.show()
    plt.savefig(f'visual/hist_ans_{filename}.png')

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('-n', '--name', default=None, type=str)

    config = ConfigParser.from_args(args)

    args = args.parse_args()
    # print(args.name)
    main(config, args.name)

#python check_age.py --c cfg/efficient_age_3_ce.json --r saved/models/efficent_age_3_ce/1027_085554/model_best.pth --n effage3ce_26_fin
#python train.py --c cfg/efficient_age_3_ce.json