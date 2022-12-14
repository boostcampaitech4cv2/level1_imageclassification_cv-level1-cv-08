# BoostCamp AI Tech4 level-1-Mask Detection Project
***
## Member๐ฅ
| [๊น๋ฒ์ค](https://github.com/quasar529) | [๊น์ฃผํฌ](https://github.com/alias26) | [๋ฐ๋ฏผ๊ท](https://github.com/zergswim) | [์ค์ฃผํ](https://github.com/OZOOOOOH) | [ํ๊ฑดํ](https://github.com/GeonHyeock) |
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
<img width="40%" src="/images/streamlit_demo.gif"/>
  

***
## Project Summary

#### ์ฃผ์ 

- ์ฃผ์ด์ง ์ด๋ฏธ์ง๋ฅผ Mask, Gender, Age๋ก ๊ตฌ์ฑ๋ 18๊ฐ์ ํด๋์ค ์ค ํ๋๋ก ์ถ์ธกํ๋ Image Classification์ด๋ค.

#### ๊ฐ์ ๋ฐ ๊ธฐ๋ํจ๊ณผ

- Week1-5๊น์ง Computer Vision์ ๋ํ ๊ต์ก์ด ์งํ๋๊ณ , Week6-7์ ๊ทธ๋์ ๋ฐฐ์ด ์ด๋ก ์ ์ ์ฉํ  ์ ์๋ Image Classification๋ํ๋ฅผ ์งํํ๋ค. 
- ์ด๋ฅผ ํตํด ๊ต์ก์์ ํ์ต ์ดํด๋๋ฅผ ํ์ธ ํด๋ณผ ์ ์๊ณ  ์ค์  ๋ฐ์ดํฐ์ ์ ์ฉํจ์ ๋ฐ๋ผ ์งํ ์ค์ธ ๊ต์ก์ ํ์์ฑ ๋ฐ ์ค์์ฑ์ ๋ค์ ํ ๋ฒ ํ์ธํ  ์ ์๋ค. ์ด๋ฅผ ์ด์ฉํด ์ผ๊ตด ์ฌ์ง๋ง์ผ๋ก Mask ์ฐฉ์ฉ ์ ๋ฌด์ ๋ฐ๋ฅด๊ฒ ์ฐฉ์ฉํ๋์ง์ ๋ํ ์๋ ํ๋ณ ๊ธฐ์ ๋ก ์ฌ์ฉ ๊ฐ๋ฅํ๋ค.


#### ๋ฐ์ดํฐ ์์ ๊ตฌ์กฐ๋

- ์ ์ฒด ์ฌ๋ ์ : 4,500๋ช
- ํ ์ฌ๋๋น ์ฌ์ง์ ๊ฐ์: 7์ฅ
(์ฐฉ์ฉ 5์ฅ, ์ด์ํ๊ฒ ์ฐฉ์ฉ(์ฝ์คํฌ, ํฑ์คํฌ) 1์ฅ, ๋ฏธ์ฐฉ์ฉ 1์ฅ)
- ์ด๋ฏธ์ง ํฌ๊ธฐ: 384 x 512(width, height)
- ์ ์ฒด ๋ฐ์ดํฐ์์ 6:4 ๋น์จ๋ก Train, Test์์ผ๋ก ์ฌ์ฉํ๊ณ  Test์์ ์ ๋ฐ ์ฉ Public, Private์์ผ๋ก ์ฌ์ฉํ๋ค.
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
ํ๋ก์ ํธ์ ์ฌ์ ๊ธฐํ ๋จ๊ณ์์๋ ํ์์ ์ํด ๋์ผํ ํํ๋ฆฟ ์ฌ์ฉ์ ๊ณํํ์ง๋ง, ๊ฐ์ธ์ ์ด๊ธฐ ์์ด๋์ด๋ฅผ ๊ตฌํํ  ์ ์๊ฒ ์์ ์๊ฒ ๋ง๋ ํํ๋ฆฟ์ ํ์ฉํ์ฌ ๊ฐ์ ์์ํ๋ค. ์ดํ Base์ฝ๋๋ฅผ ์ ์ ํ์ฌ Black ํฌ๋งทํ์ ์ฌ์ฉํ์ฌ ํ์์ ์งํํ๊ธฐ๋ก ํ์๋ค.

**[2022.10.24 ~ 2022.11.01]**  
Base์ฝ๋ ์ ์ ์ผ(10.26)๊น์ง Base์ฝ๋๋ฅผ ๋ฏธ์์ฑํ๊ฑฐ๋ ์ถ๊ฐ ์คํ์ ์ํ๋ ์ธ์์ด ์๊น์ ๋ฐ๋ผ Base์ฝ๋ ์ ์ ๋ ์ง๋ฅผ ๋ฏธ๋ฃจ๊ฒ ๋์๋ค.

*๊ฐ์ธ๋ณ ์ฃผ์ ์คํ ์ ๋ต*
- ๊น๋ฒ์ค(Loss,Optimizer,Augmentation), 
- ๊น์ฃผํฌ(Loss, Augmentation), 
- ๋ฐ๋ฏผ๊ท((๋ณ๋์งํ) kfold), 
- ์ค์ฃผํ(TRACER), 
- ํ๊ฑดํ(CLIP model)

**[2022.11.02 ~ 2022.11.03]**  
๋ชจ๋์ ์๊ฒฌ์ ์กด์คํ์ฌ ํ์์ ํ  ํ์๊ณผ ๊ฐ์ธ์ด ํ๋ ์คํ์ ๋ง๋ฌด๋ฆฌํ  ํ์์ ๋๋ ์ ๋จ์ ๊ธฐ๊ฐ ๋์ ํ๋ก์ ํธ๋ฅผ ์งํํ๊ฒ ๋์๋ค.
ํ์์ Git-Flow๋ฅผ ์ด์ฉํ๊ณ  Base์ฝ๋๋ฅผ ์ ํ ํ, ํ์ํ ๊ธฐ๋ฅ์ ๊ตฌํํด ์ถ๊ฐํ๋ค.
๊ทธ ํ ๊ฐ์ธ์ ๊ฐ์ค์ ๋ฐ๋ผ ์คํ์ ์งํํ๊ณ  ๊ฐ์ฅ ๋ฐ์ด๋ ์ฑ๋ฅ(F1-Score)์ ๋ณด์ธ ๊ฒฐ๊ณผ๋ฅผ ์ต์ข ์ ์ถํ๋ค.  
์ต์ข ์์ : 12th / 20
***

## Features

**feat-multilabel** : 18๊ฐ์ Class๋ฅผ ๋ถ๋ฅํ๋ ๋ฌธ์ ๋ฅผ 3๊ฐ์ Multi-Label Class๋ก ๋ถ๋ฅํ๋ ๋ฌธ์ ๋ก ์นํ, Multi loss ์ถ๊ฐ

**feat-TRACER** : ๋ฐฐ๊ฒฝ ์ด๋ฏธ์ง ์ ๊ฑฐํด์ฃผ๋ ๊ธฐ๋ฅ ์ถ๊ฐ

**feat-wandb** : wandb, tqdm ์ฐ๋

**feat-yml** : pytorch metric learning์ loss ์ถ๊ฐ

**feat-strmlit** : streamlit demo 

***
## Result
#### EDA
- ๋ฐ์ดํฐ ํด๋์ค์ ๋ถ๊ท ํ์ด ์๋ ์ ์ ํ์(๋์ด, ์ฑ๋ณ)
- ๋ฐ์ดํฐ ํ์์ ์ ์ฒด์ ์ผ๋ก ํต์ผ๋์ด ์์(์ผ๊ตด ์์น)

#### ๋ฐ์ดํฐ ์ ์ฒ๋ฆฌ
- ์ฃผ์ด์ง Train ๋ฐ์ดํฐ๋ฅผ 9:1 ๋น์จ๋ก Train, Validation์์ผ๋ก
๋ชจ๋ธ ํ์ต์ด ์ ๋๊ณ  ์๋์ง์ ๋ํด ๊ฒ์ฆํ๊ธฐ ์ํด์ ์ฌ์ฉ
- ๋ฐ์ดํฐ๊ฐ ์ฌ๋ 1๋ช๊ณผ ๋ฐฐ๊ฒฝ๋ง ์๋ ๋จ์ํ ๊ตฌ์กฐ์ด๊ธฐ ๋๋ฌธ์
์ฒ์์๋ Face Detection ๋ชจ๋ธ์ ์ด์ฉํด ์ผ๊ตด์ Crop ํ
ํ์ต์ ์งํํ๋ ค ํ์ผ๋ Mask๋ฅผ ์ด ์ผ๊ตด์ ๋ชจ๋ธ์ด ์ ํ์งํ์ง ๋ชปํ  ๊ฑฐ๋ผ ์์ํด ๋ค๋ฅธ ๋ชจ๋ธ์ ์ฌ์ฉ.
- RGB-Salient Object Detection Task์์ TRACER ๋ชจ๋ธ์ ์ฌ์ฉํ๊ธฐ๋ก ๊ฒฐ์ . ์ด ๋ชจ๋ธ์ ์ฃผ์ด์ง ์ด๋ฏธ์ง์์ Salient(ํต์ฌ์ ์ธ) Object๋ฅผ ์ฐพ์์ฃผ๊ณ  Box ํํ๊ฐ ์๋ Segmentation ํํ๋ก output์ด ๋์จ๋ค. ํด๋น ๋ชจ๋ธ์ด ์ฌ๋๋ง segmentationํด์ค ๊ฒ์ผ๋ก ์์ํด ์ฌ์ฉํ๊ณ  ๊ฒฐ๊ณผ๊ฐ ๊ด์ฐฎ์์ ํ์ตํ  ๋ ์ฌ์ฉํ๋ค.
 
#### Data Augmentation
- Resize๋ ๋ชจ๋ธ์ ํ์ต๊ณผ ํ์ต์๊ฐ๋จ์ถ
- ShiftScaleRotate๋ ์ผ๋ฐํ, ๋ฐ์ดํฐ Label์ ํน์ง์ ์ ์งํ๊ธฐ ์ํด์ ์ฝํ๊ฒ ๋ฃ์ด์ค(shift,scale 10%,rotation 5ยฐ)
- RandomBrightnessContrast๋ Mask๊ฐ ํฐ์ ๋ง๊ณ  ๋ค๋ฅธ ์๋ ์๊ธฐ ๋๋ฌธ์ ์ผ๋ฐํ๋ฅผ ์ํด์ ๋ฐ๊ธฐ์ ๋์กฐ๋ฅผ ์์ฃผ ์ฝํ๊ฒ ๋ฃ์ด์คฌ๋ค.(๊ฐ๊ฐ 20%)
- HorizontalFlip์ ์ข์ฐ ๋ฐ์ ์ด๊ธฐ ๋๋ฌธ์ ๋จ์ ๋ฐ์ดํฐ ์ฆ๊ฐ์ ์ํด ์ฌ์ฉ
- CoarseDropout์ cut-out๊ธฐ๋ฒ์ผ๋ก Overfitting ๋ฐฉ์ง๋ฅผ ์ํด ์ฌ์ฉ
 
#### ๋ชจ๋ธ ๊ฐ์
- ๊ธฐ๋ณธ ๋ชจ๋ธ์ ๋น ๋ฅธ ์คํ๊ณผ ์์ด๋์ด ๊ตฌํ์ ์ํด์ Efficientnet ๊ณ์ด ๋ชจ๋ธ์ ์ฌ์ฉํ๊ธฐ๋ก ๊ฒฐ์ .
- Backbone์ผ๋ก Efficientnetv2_s๋ฅผ ์ด์ฉํ์ฌ ๊ณตํต๋ feature๋ฅผ ์ถ์ถํ ํ, 3-Way fc๋ฅผ ํต๊ณผํ์ฌ Mask, Gender, Age ๊ฐ๊ฐ์ Class๋ฅผ ๋ถ๋ฅ.
- Multi-Label ๋ฌธ์ ๋ฅผ ํด๊ฒฐํ๊ธฐ ์ํ์ฌ ๊ฐ๊ฐ์ Class์ ๋ํ์ฌ Cross-entropy Loss๋ฅผ ์ค์ ํ์ฌ ๋ํ๋ ๋ฐฉ์์ผ๋ก ์ค์ ํ์๊ณ  ์ถ๊ฐ๋ก Data Imbalance๋ฅผ ๊ณ ๋ คํ์ฌ Loss์ Weight๋ฅผ ์ถ๊ฐ.

#### ์์ฐ ๊ฒฐ๊ณผ
|<img width="100%" src="/images/wandb_loss.png"/>|<img width="100%" src="/images/wandb_accuracy.png"/>|<img width="100%" src="/images/wandb_f1score.png"/>|
|----|----|----|

|Submit/F1 Score|Submit/Accuracy|
|----|----|
|<div style="text-align: center">0.7040</div>|<div style="text-align: center">75.0635</div>|

***
## Conclusion
#### ์ํ ์ ๋ค
- ํ ๋จ์๋ก ์งํํ๊ธฐ ์  ๊ฐ์ ์๊ฐํ ๊ฐ์ค๊ณผ ๋์ํ๋ ์ ๋ต์ ๋ฐ๋ผ ๋ค์ํ ์คํ์ ์งํํด ์ด์ ์ ์๋ํ์ง ๋ชปํ ๋ฐฉ๋ฒ์ ์ ์ฉํด ๋ณผ ์ ์์๋ค.
- ํ์ผ๋ก ์งํ ํ์๋ git flow๋ฅผ ์ด์ฉํ ํ์์ ๊ฒฝํ ํ  ์ ์์๋ค.
- Template์ ์ฌ์ฉํด ์ตํ์ผ๋ก์ ๋ค๋ฅธ ๋ํ ๋ฐ ํ๋ซํผ์์๋ ํ์ฉํ  ์ ์๊ฒ ๋์๋ค.

#### ์์ฌ์ ๋ ์ ๋ค:
- git ์ฌ์ฉ ์ Issue๋ณ๋ก ๊ตฌ๋ถํด ์ฌ์ฉ ๋ชปํ๋ค.
- ์๊ฐ ๋ถ์กฑ์ผ๋ก Ensemble ์๋ ๋ชปํ๋ค.
- ๊ฐ์ธ์ ์ผ๋ก ์คํ์ ํ  ๋ ๋ฌธ์ ๊ฐ ์๊ธธ ๊ฒฝ์ฐ ํผ์์ ํด๊ฒฐํด์ผ ํด์ ๋ง์ ๋ธ๋ ฅ๊ณผ ์๊ฐ์ด ์์๋์ด ๋ค์ํ ์๋๋ฅผ ํ  ์ฌ์ ๊ฐ ๋ถ์กฑํ๋ค.

#### ํ๋ก์ ํธ๋ฅผ ํตํด ๋ฐฐ์ด์ :
- pytorch-template ๊ตฌ์กฐ ํ์, ๊ธฐ๋ฅ ์ถ๊ฐ ๊ตฌํ์ผ๋ก ์ถํ์ ํ์ฉ ๊ฐ๋ฅ.
- wandb๋ฅผ ํตํ ์คํ ํผ๋๋ฐฑ.
- Git Flow๋ฅผ ํตํ ํ์ ํ๋ก์ธ์ค ์ฒดํ ๋ฐ ์ ์ฉ ๊ฐ๋ฅ.
- ์ฃผ์ด์ง ๋ฐ์ดํฐ ๋ถ์์ ์ฒ ์ ํ ์งํ ํ ์ด์ ๋์ํ๋ ์ ๋ต์ ์ธ์์ผ ํ๋ค.
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
  โ
  โโโ train.py - main script to start training
  โโโ test.py - evaluation of trained model
  โ
  โโโ config.json - holds configuration for training
  โโโ parse_config.py - class to handle config file and cli options
  โ
  โโโ new_project.py - initialize new project with template files
  โ
  โโโ TRACER/ - face detection modules
  โ
  โโโ Base/ - abstract Base classes
  โ   โโโ Base_data_loader.py
  โ   โโโ Base_model.py
  โ   โโโ Base_trainer.py
  โ
  โโโ data_loader/ - anything about data loading goes here
  โ   โโโ data_loaders.py
  โ   โโโ transform.py
  โ
  โโโ data/ - default directory for storing input data
  โ
  โโโ model/ - models, losses, and metrics
  โ   โโโ model.py
  โ   โโโ metric.py
  โ   โโโ loss.py
  โ
  โโโ saved/
  โ   โโโ models/ - trained models are saved here
  โ   โโโ log/ - default logdir for tensorboard and logging output
  โ
  โโโ trainer/ - trainers
  โ   โโโ trainer.py
  โ
  โโโ logger/ - module for tensorboard visualization and logging
  โ   โโโ visualization.py
  โ   โโโ logger.py
  โ   โโโ logger_config.json
  โ  
  โโโ utils/ - small utility functions
  โ   โโโ util.py
  โ
  โโโ submit/ - submission are saved here 
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
