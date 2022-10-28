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
        # Normalize((0.1307,), (0.3081,)),
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

    def decoder(no):
        mask = no//6
        gender = no//3%2
        age = no%3
        return mask, gender, age

    # for i, label in enumerate(test_sets.classes):
    plt.figure(figsize=(16,16))
    for i, label in enumerate(range(18)):
        idx = np.where(actual==i)
        # print(type(idx), len(idx), idx)
        plt.scatter(cluster[idx,0], cluster[idx,1], marker='.', label=label)
    plt.title('label')
    plt.axis('off')
    plt.legend()
    plt.show()
    plt.savefig(f'visual/tSNE_org_{filename}.png')

    plt.figure(figsize=(16,16))
    mask_list = [[], [], []]
    for i, label in enumerate(range(18)):
        idx = np.where(actual==i)
        mask, gender, age = decoder(label)
        if mask == 0: mask_list[0].extend(idx[0])
        if mask == 1: mask_list[1].extend(idx[0])
        if mask == 2: mask_list[2].extend(idx[0])

    for i, idxs in enumerate(mask_list):
        # print("mask:",type(idxs), len(idxs), idxs)
        plt.scatter(cluster[idxs,0], cluster[idxs,1], marker='.', label=i)
    plt.title('mask')
    plt.axis('off')
    plt.legend()
    plt.show()
    plt.savefig(f'visual/tSNE_mask_{filename}.png')

    plt.figure(figsize=(16,16))
    gender_list = [[], []]
    for i, label in enumerate(range(18)):
        idx = np.where(actual==i)
        mask, gender, age = decoder(label)
        if gender == 0: gender_list[0].extend(idx[0])
        if gender == 1: gender_list[1].extend(idx[0])

    for i, idxs in enumerate(gender_list):
        plt.scatter(cluster[idxs,0], cluster[idxs,1], marker='.', label=i)
    plt.title('gender')
    plt.axis('off')
    plt.legend()
    plt.show()
    plt.savefig(f'visual/tSNE_gender_{filename}.png')

    plt.figure(figsize=(16,16))
    age_list = [[], [], []]
    for i, label in enumerate(range(18)):
        idx = np.where(actual==i)
        mask, gender, age = decoder(label)
        if age == 0: age_list[0].extend(idx[0])
        if age == 1: age_list[1].extend(idx[0])
        if age == 2: age_list[2].extend(idx[0])

    for i, idxs in enumerate(age_list):
        plt.scatter(cluster[idxs,0], cluster[idxs,1], marker='.', label=i)
    plt.title('age')
    plt.axis('off')
    plt.legend()
    plt.show()
    plt.savefig(f'visual/tSNE_age_{filename}.png')


    submission['ans'] = all_predictions

    # 제출할 파일을 저장합니다.
    # submission.to_csv(os.path.join(test_dir, 'submission.csv'), index=False)
    submission.to_csv(f'submit/submission_{filename}.csv', index=False)
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

# 0.01 stepLR, batch 64, resnet50(pre), epoch 3, ce
#{'loss': 0.35128293634111446, 'accuracy': 0.8778835978835979, 'top_k_acc': 0.9926455026455027}
#python submit.py --r saved/models/L1/1025_054359/model_best.pth

# 0.01 stepLR, batch 64, resnet50(pre), epoch 10, ce, agmentation(RandomHorizontalFlip, RandomAffine)
#{'loss': 0.0074951862377575324, 'accuracy': 0.9975132275132275, 'top_k_acc': 1.0}
#python submit.py --r saved/models/L11/1025_091936/model_best.pth

# 0.01 stepLR, batch 64, resnet50(pre), epoch 20, ce, agmentation(RandomHorizontalFlip, RandomAffine)
#python submit.py --r saved/models/L11/1025_112910/model_best.pth
#{'loss': 0.0014902268164537393, 'accuracy': 0.9996825396825397, 'top_k_acc': 1.0}

# 0.01 stepLR, batch 64, resnet50(pre), epoch 30, ce, agmentation(RandomHorizontalFlip, RandomAffine)
#python submit.py --r saved/models/L11/1025_235220/model_best.pth
#python train.py --c cfg/l11.json --r saved/models/L11/1025_235220/model_best.pth

# 0.01 stepLR, batch 64, resnet50(pre), epoch 38, ce, agmentation(RandomHorizontalFlip, RandomAffine)
#     epoch          : 38
#     loss           : 0.001544304876334447
#     accuracy       : 0.9994548872180452
#     top_k_acc      : 1.0
#     val_loss       : 0.004491845703159925
#     val_accuracy   : 0.9989583333333333
#     val_top_k_acc  : 1.0
# Validation performance didn't improve for 10 epochs. Training stops.
#python submit.py --r saved/models/L11/1026_030706/model_best.pth --n res_38_fin

#     epoch          : 22
#     loss           : 0.0010055753522542382
#     accuracy       : 0.9996475563909775
#     top_k_acc      : 1.0
#     val_loss       : 0.009862332145303299
#     val_accuracy   : 0.9973958333333334
#     val_top_k_acc  : 1.0
# Validation performance didn't improve for 10 epochs. Training stops.    
#python submit.py --c cfg/efficient.json --r saved/models/efficent/1027_011536/model_best.pth --n eff_22_fin