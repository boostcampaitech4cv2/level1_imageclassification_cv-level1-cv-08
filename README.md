# BoostCamp AI Tech4 level-1-Mask Detection Project
***
## Member🔥
| [김범준](https://github.com/quasar529) | [김주희](https://github.com/alias26) | [박민규](https://github.com/zergswim) | [오주헌](https://github.com/OZOOOOOH) | [허건혁](https://github.com/GeonHyeock) |
| :-: | :-: | :-: | :-: | :-: |
| <img src="https://avatars.githubusercontent.com/quasar529" width="100"> | <img src="https://avatars.githubusercontent.com/alias26" width="100"> | <img src="https://avatars.githubusercontent.com/zergswim" width="100"> | <img src="https://avatars.githubusercontent.com/OZOOOOOH" width="100"> | <img src="https://avatars.githubusercontent.com/GeonHyeock" width="100"> |
***
## Index
* [Demo Video](#demo-video)
* [Project Summary](#project-summary)
* [Requirements](#requirements)
* [Procedures](#procedures)
* [Features](#features)
* [Result](#result)
* [Conclusion](#Conclusion)  
***
## Demo Video
<img width="100%" src="/images/streamlit_demo.gif"/>
  

***
## Project Summary

#### 주제

- 주어진 이미지를 Mask, Gender, Age로 구성된 18개의 클래스 중 하나로 추측하는 Image Classification이다.

#### 개요 및 기대효과

- Week1-5까지 Computer Vision에 대한 교육이 진행됐고, Week6-7에 그동안 배운 이론을 적용할 수 있는 Image Classification대회를 진행했다. 
- 이를 통해 교육생의 학습 이해도를 확인 해볼 수 있고 실제 데이터에 적용함에 따라 진행 중인 교육의 필요성 및 중요성을 다시 한 번 확인할 수 있다. 이를 이용해 얼굴 사진만으로 Mask 착용 유무와 바르게 착용했는지에 대한 자동 판별 기술로 사용 가능하다.


#### 데이터 셋의 구조도

- 전체 사람 수 : 4,500명
- 한 사람당 사진의 개수: 7장
(착용 5장, 이상하게 착용(코스크, 턱스크) 1장, 미착용 1장)
- 이미지 크기: 384 x 512(width, height)
- 전체 데이터셋은 6:4 비율로 Train, Test셋으로 사용하고 Test셋은 절반 씩 Public, Private셋으로 사용한다.
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
## Procedures

**[2022.10.20 ~ 2022.10.21]**  
프로젝트의 사전기획 단계에서는 협업을 위해 동일한 템플릿 사용을 계획했지만, 개인의 초기 아이디어를 구현할 수 있게 자신에게 맞는 템플릿을 활용하여 각자 작업했다. 이후 Base코드를 선정하여 Black 포맷팅을 사용하여 협업을 진행하기로 하였다.

**[2022.10.24 ~ 2022.11.01]**  
Base코드 선정일(10.26)까지 Base코드를 미완성하거나 추가 실험을 원하는 인원이 생김에 따라 Base코드 선정날짜를 미루게 되었다.

*개인별 주요 실험 전략*
- 김범준(Loss,Optimizer,Augmentation), 
- 김주희(Loss, Augmentation), 
- 박민규((별도진행) kfold), 
- 오주헌(TRACER), 
- 허건혁(CLIP model)

**[2022.11.02 ~ 2022.11.03]**  
모두의 의견을 존중하여 협업을 할 팀원과 개인이 하던 실험을 마무리할 팀원을 나눠서 남은 기간 동안 프로젝트를 진행하게 되었다.
협업은 Git-Flow를 이용했고 Base코드를 정한 후, 필요한 기능을 구현해 추가했다.
그 후 개인의 가설에 따라 실험을 진행했고 가장 뛰어난 성능(F1-Score)을 보인 결과를 최종 제출했다.  
최종 순위 : 12th / 20
***

## Features

**feat-multilabel** : 18개의 Class를 분류하는 문제를 3개의 Multi-Label Class로 분류하는 문제로 치환, Multi loss 추가

**feat-TRACER** : 배경 이미지 제거해주는 기능 추가

**feat-wandb** : wandb, tqdm 연동

**feat-yml** : pytorch metric learning의 loss 추가

***
## Result
#### EDA
- 데이터 클래스에 불균형이 있는 점을 파악(나이, 성별)
- 데이터 형식은 전체적으로 통일되어 있음(얼굴 위치)

#### 데이터 전처리
- 주어진 Train 데이터를 9:1 비율로 Train, Validation셋으로
모델 학습이 잘 되고 있는지에 대해 검증하기 위해서 사용
- 데이터가 사람 1명과 배경만 있는 단순한 구조이기 때문에
처음에는 Face Detection 모델을 이용해 얼굴을 Crop 후
학습을 진행하려 했으나 Mask를 쓴 얼굴은 모델이 잘 탐지하지 못할 거라 예상해 다른 모델을 사용.
- RGB-Salient Object Detection Task에서 TRACER 모델을 사용하기로 결정. 이 모델은 주어진 이미지에서 Salient(핵심적인) Object를 찾아주고 Box 형태가 아닌 Segmentation 형태로 output이 나온다. 해당 모델이 사람만 segmentation해줄 것으로 예상해 사용했고 결과가 괜찮아서 학습할 때 사용했다.
 
#### Data Augmentation
- Resize는 모델의 학습과 학습시간단축
- ShiftScaleRotate는 일반화, 데이터 Label의 특징을 유지하기 위해서 약하게 넣어줌(shift,scale 10%,rotation 5°)
- RandomBrightnessContrast는 Mask가 흰색 말고 다른 색도 있기 때문에 일반화를 위해서 밝기와 대조를 아주 약하게 넣어줬다.(각각 20%)
- HorizontalFlip은 좌우 반전이기 때문에 단순 데이터 증강을 위해 사용
- CoarseDropout은 cut-out기법으로 Overfitting 방지를 위해 사용
 
#### 모델 개요
- 기본 모델은 빠른 실험과 아이디어 구현을 위해서 Efficientnet 계열 모델을 사용하기로 결정.
- Backbone으로 Efficientnetv2_s를 이용하여 공통된 feature를 추출한 후, 3-Way fc를 통과하여 Mask, Gender, Age 각각의 Class를 분류.
- Multi-Label 문제를 해결하기 위하여 각각의 Class에 대하여 Cross-entropy Loss를 설정하여 더하는 방식으로 설정하였고 추가로 Data Imbalance를 고려하여 Loss에 Weight를 추가.

#### 시연 결과
|<img width="100%" src="/images/wandb_loss.png"/>|<img width="100%" src="/images/wandb_accuracy.png"/>|<img width="100%" src="/images/wandb_f1score.png"/>|
|----|----|----|

|Submit/F1 Score|Submit/Accuracy|
|----|----|
|<div style="text-align: center">0.7040</div>|<div style="text-align: center">75.0635</div>|

***
## Conclusion
#### 잘한 점들
- 팀 단위로 진행하기 전 각자 생각한 가설과 대응하는 전략에 따라 다양한 실험을 진행해 이전에 시도하지 못한 방법을 적용해 볼 수 있었다.
- 팀으로 진행 후에는 git flow를 이용한 협업을 경험 할 수 있었다.
- Template을 사용해 익힘으로서 다른 대회 및 플랫폼에서도 활용할 수 있게 되었다.

#### 아쉬웠던 점들:
- git 사용 시 Issue별로 구분해 사용 못했다.
- 시간 부족으로 Ensemble 시도 못했다.
- 개인적으로 실험을 할 때 문제가 생길 경우 혼자서 해결해야 해서 많은 노력과 시간이 소요되어 다양한 시도를 할 여유가 부족했다.

#### 프로젝트를 통해 배운점:
- pytorch-template 구조 파악, 기능 추가 구현으로 추후에 활용 가능.
- wandb를 통한 실험 피드백.
- Git Flow를 통한 협업 프로세스 체험 및 적용 가능.
- 주어진 데이터 분석을 철저히 진행 후 이에 대응하는 전략을 세워야 한다.
***
## Requirements
* Python >= 3.8 (3.8 recommended)
* PyTorch >= 1.2
* albumentations>=1.3.0
* tqdm
* wandb
* pytorch_metric_learning
* sklearn
* timm
* tensorboard >= 1.14
* Black == 22.10.0
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
  ├── Base/ - abstract Base classes
  │   ├── Base_data_loader.py
  │   ├── Base_model.py
  │   └── Base_trainer.py
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
    "name": "Mask_Base",                    // training session name
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

### Train, Test using config files
Modify `config.json` by your setting:

  ```
  python train.py --config config.json
  
  python test.py --config config.json --resume "Your Checkpoint Path"
  ```

### Streamlit Prediction
Run and Check your Mask Classification Prediction result:

```
streamlit run app.py --server.port="Your Port Number"
```

## License
[This project is licensed under the MIT License. See  LICENSE for more details](https://github.com/victoresque/pytorch-template)
