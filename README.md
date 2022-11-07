# BoostCamp AI Tech4 level-1-Mask Detection Project

## Index
* [Project Summary](#project-summary)
* [Requirements](#requirements)
* [Procedures](#procedures)
* [Features](#features)
* [Result](#result)
* [Conclusion](#Conclusion)
## Project Summary

##### 주제

주어진 이미지 데이터를 Mask(Correct, Incorrect, Not Wear), Gender(Male,Female), Age(Young, Middle, Old)를 각각 구분해 18개 클래스 중 하나로 추측한다.
이를 이용해 카메라에 찍힌 사람 얼굴만으로 이 사람이 Mask 착용 유무와 제대로 착용했는지 자동 판별 시스템의 기술로 사용 가능하다.

#### 개요 및 기대효과

Week1~5까지 Computer Vision에 대한 전반적인 교육이 진행됐고, Week6~7에 그동안 배운 이론을 활용할 수 있는 Image Classification대회를 진행했다. 
이를 통해 교육생의 학습 이해도를 확인 해볼 수 있고 실제 데이터에 적용함에 따라 진행 중인 교육의 필요성 및 중요성을 다시 한 번 확인할 수 있다.

#### 데이터 셋의 구조도

- 전체 사람 수 : 4,500명
- 한 사람당 사진의 개수: 7 (착용 5장, 이상하게 착용(코스크, 턱스크) 1장, 미착용 1장)
- 이미지 크기: 384 x 512(width, height)
- 전체 데이터셋 중에서 60%는 학습 데이터셋으로 활용한다.
나머지 40%의 데이터셋은 테스트셋으로 활용되고 각각 절반으로 나눠 public, private 데이터셋으로 사용된다.

#### Data Description

|Class|Mask|Gender|AGE|
|---|---|---|---|
|0|Wear|Male|< 30|
|1|Wear|Male|>= 30 and < 60|
|2|Wear|Male|>= 60|
|3|Wear|Female|< 30|
|4|Wear|Female|>= 30 and < 60|
|5|Wear|Female|>= 60|
|6|Incorrect|Male|< 30|
|7|Incorrect|Male|>= 30 and < 60|
|8|Incorrect|Male|>= 60|
|9|Incorrect|Female|< 30|
|10|Incorrect|Female|>= 30 and < 60|
|11|Incorrect|Female|>= 60|
|12|Not Wear|Male|< 30|
|13|Not Wear|Male|>= 30 and < 60|
|14|Not Wear|Male|>= 60|
|15|Not Wear|Female|< 30|
|16|Not Wear|Female|>= 30 and < 60|
|17|Not Wear|Female|>= 60|
***
## Requirements
* Python >= 3.8 (3.8 recommended)
* PyTorch >= 1.2 (1.2 recommended)
* tqdm
* wandb
* tensorboard >= 1.14 (see [Tensorboard Visualization](#tensorboard-visualization))
***
## Procedures

**[2022.10.20 ~ 2022.10.21]**
프로젝트의 사전기획 단계에서는 협업을 위하여 통일된 템플릿을 지향하나, 개인의 초기 아이디어를 구현할 수 있게 자신에게 맞는 템플릿을 활용하여 각자 자유롭게 base 작업을 진행한다. 이후 base코드를 선정하여 이후 협업을 진행하기로 하였다.
 
**[2022.10.24 ~ 2022.10.26]**
프로젝트가 시작되고 base코드 선정날짜인 26일 base코드를 완성하지 못 하거나 추가 실험을 원하는 인원이 생김에 따라 base코드 선정날짜를 미루게 되었다. 
 
**[2022.10.26 ~ 2022.11.01]**
base코드를 선정함에 있어 의견이 일치하지 않아 개인의 아이디어를 구술로 공유를 하였고 작업은 개인별로 진행되었다.
 
**[2022.11.02 ~ 2022.11.03]**
base코드를 선정하여 협업을 할 인원과 개인이 진행하던 프로젝트를 마무리 할 인원 각자의 의견을 존중하여 나누어 프로젝트를 마무리 하게 되었다.
협업을 하는 과정에는 각자 작성하였던 코드 중 병합하였을때 필요로 하는 코드와 새롭게 작성해야하는 코드들을 git flow전략을 바탕으로 선정된 base코드를 기준으로 병합하였다.
***
## Result
#### EDA
- 데이터 클래스에 불균형이 있는 점을 파악(나이, 성별)
- 데이터 형식은 전체적으로 통일되어 있음(얼굴 위치)

#### 데이터 전처리
- 주어진 train 데이터를 9:1 비율로 train, validation셋으로
모델 학습이 잘 되고 있는지에 대해 검증하기 위해서 사용
- 데이터가 사람 1명과 배경만 있는 단순한 구조이기 때문에
처음에는 Face Detection 모델을 이용해 얼굴을 crop 후
학습을 진행하려 했으나 Mask를 쓴 얼굴은 모델이 잘 탐지하지 못할 거라 예상해 다른 모델을 사용.
- RGB-Salient Object Detection task에서 TRACER 모델을 사용하기로 결정. 이 모델은 주어진 이미지에서 Salient(핵심적인) Object를 찾아주고 Box 형태가 아닌 Segmentation 형태로 output이 나온다. 해당 모델이 사람만 segmentation해줄 것으로 예상해 사용했고 결과가 괜찮아서 학습할 때 사용했다.
 
#### Data Augmentation
- Resize는 모델의 학습과 학습시간단축
- ShiftScaleRotate는 일반화, 데이터 label의 특징을 유지하기 위해서 약하게 넣어줌(shift,scale 10%,rotation 5°)
- RandomBrightnessContrast는 Mask가 흰색 말고 다른 색도 있기 때문에 일반화를 위해서 밝기와 대조를 아주 약하게 넣어줬다.(각각 20%)
- HorizontalFlip은 좌우 반전이기 때문에 단순 데이터 증강을 위해 사용
- CoarseDropout은 cutout기법으로 overfitting 방지를 위해 사용
 
#### 모델 개요
- 기본 모델은 빠른 실험과 아이디어 구현을 위해서 Efficientnet 계열 모델을 사용하기로 결정
- Backbone으로 efficientnetv2_s를 이용하여 공통된 feature를 추출한 후, 3way fc를 통과하여 mask, gender, age 각각의 class를 분류
- multi label 문제를 해결하기 위하여 각각의 class에 대하여 cross entropy loss를 설정하여 더하는 방식으로 설정하였고 추가로 data unbalance를 고려하여 loss에 weight를 추가

#### 시연 결과
![]()
![]()
![]()
![]()

|Submit/F1 Score|Submit/Accuracy|
|----|----|
|<div style="text-align: center">0.7040</div>|<div style="text-align: center">75.0635</div>|


## Conclusion
#### 잘한 점들
- 팀 단위로 진행하기 전 각자 생각한 가설과 대응하는 전략에 따라 다양한 실험을 진행해 이전에 시도하지 못한 방법을 적용해 볼 수 있었다.도
- 팀으로 진행 후에는 git flow를 이용한 협업을 경험 할 수 있었다.
- template을 사용해 익힘으로서 다른 대회 및 플랫폼에서도 활용할 수 있게 되었다.

#### 아쉬웠던 점들:
- git 사용 시 Issue별로 구분해 사용 못했다.
- 시간 부족으로 ensemble 시도 못했다.
- 개인적으로 실험을 할 때 문제가 생길 경우 혼자서 해결해야 해서 많은 노력과 시간이 소요되어 다양한 시도를 할 여유가 부족했다.

#### 프로젝트를 통해 배운점:
- pytorch-template 구조 파악, 기능 추가 구현으로 추후에 활용 가능.
- wandb를 통한 실험 피드백.
- Git Flow를 통한 협업 프로세스 체험 및 적용 가능.
- 주어진 데이터 분석을 철저히 진행 후 이에 대응하는 전략을 세워야 한다.
***

## Folder Structure
  ```
  level1-imageclassification-cv-08/
  │
  ├── train.py - main script to start training
  ├── test.py - evaluation of trained model
  │
  ├── config.json - holds configuration for training
  ├── parse_config.py - class to handle config file and cli options
  │
  ├── new_project.py - initialize new project with template files
  │
  ├── TRACER/ - face detection modules
  │
  ├── base/ - abstract base classes
  │   ├── base_data_loader.py
  │   ├── base_model.py
  │   └── base_trainer.py
  │
  ├── data_loader/ - anything about data loading goes here
  │   ├── data_loaders.py
  │   └── transform.py
  │
  ├── data/ - default directory for storing input data
  │
  ├── model/ - models, losses, and metrics
  │   ├── model.py
  │   ├── metric.py
  │   └── loss.py
  │
  ├── saved/
  │   ├── models/ - trained models are saved here
  │   └── log/ - default logdir for tensorboard and logging output
  │
  ├── trainer/ - trainers
  │   └── trainer.py
  │
  ├── logger/ - module for tensorboard visualization and logging
  │   ├── visualization.py
  │   ├── logger.py
  │   └── logger_config.json
  │  
  │── utils/ - small utility functions
  │   └── util.py
  │
  └── submit/ - submission are saved here 
  ```
  ***
  ### Config file format
Config files are in `.json` format:
```javascript
{
    "name": "Mask_base",                    // training session name
    "n_gpu": 1,                             // number of GPUs to use for training.
    "arch": {
        "type": "TimmModelMulti",
        "args": {
            "model_name": "efficientnetv2_rw_s",  // name of model architecture to train
            "pretrained": true
        }
    },
    "data_loader": {
        "type": "setup",                // selecting data loader
        "args": {
            "stage": "train",           // selecting the stage between train and eval
            "input_size": 240,          // image resize size
            "batch_size": 16,           // batch size
            "num_workers": 4            // number of cpu processes to be used for data loading
        }
    },
    "optimizer": {
        "type": "Adam",                 // optimizer type
        "args": {
            "lr": 0.0001,               // learning rate
            "weight_decay": 0,          // weight decay
            "amsgrad": true
        }
    },
    "loss": "all_loss",                 // loss type
    "loss_name": [
        [
            "focal",
            [
                1.4,            // multi_loss CE weight
                7.0,
                7.0
            ],
            0.375
        ],
        [
            "focal",
            [
                1.6285,
                2.5912
            ],
            0.25
        ],
        [
            "focal",
            [
                2.8723,
                3.9589,
                4.5075,
                14.0625,
                14.0625,
                28.4211
            ],
            0.375
        ]
    ],
    "metrics": [
        "accuracy", "f1" // list of metrics to evaluate
    ],
    "lr_scheduler": {
        "type": "StepLR", // learning rate scheduler type
        "args": {
            "step_size": 5,
            "gamma": 0.1
        }
    },
    "trainer": {
        "epochs": 50,             // number of training epochs
        "save_dir": "saved/",     // checkpoints are saved in save_dir/models/name
        "save_period": 1,         // save checkpoints every save_freq epochs
        "verbosity": 2,           // 0: quiet, 1: per epoch, 2: full
        "monitor": "min val/loss",// mode and metric for model performance monitoring. set 'off' to disable.
        "early_stop": 3,          // number of epochs to wait before early stop. set 0 to disable.
        "tensorboard": false     // enable tensorboard visualization
    },
    "wandb": true,   // enable wandb logging
    "visualize": false, // enable visualization of wandb visualization
    "submit_dir": "/submit", // submission csv directory
    "info_dir": "/input/data/eval"// eval dataset directory
}
```

**Add addional configurations if you need.**

### Using config files
Modify the configurations in `.json` config files, then run:

  ```
  python train.py --config config.json
  ```

### Resuming from checkpoints
You can resume from a previously saved checkpoint by:

  ```
  python train.py --resume path/to/checkpoint
  ```
### Testing 
## License
[This project is licensed under the MIT License. See  LICENSE for more details](https://github.com/victoresque/pytorch-template)
