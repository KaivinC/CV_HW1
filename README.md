# Kaggle-HW1

Code for 10th place solution in Kaggle Human CS_T0828_HW1 Challenge.

### Reproducing Submission
To reproduct my submission without retrainig, do the following steps:
1. [Installation](#installation)
2. [Download Official Image](#download-official-image)
3. [Download Pretrained models](#pretrained-models)
4. [Inference](#inference)
5. [Make Submission](#make-submission)

### Installation
```bash
python -m pip install -r requirements.txt
```

### Dataset Prepare 
#### Prepare Images
After downloading and converting images, the data directory is structured as:
```bash
datasets
  +- 
  |  +- training_data
  |  +- testing_data
  |  +- training_labels.csv
  |  +- anno
```

Note: The `label_num` in annotations starts from 1 rather than 0.

#### Download Official Image
Download and extract train.zip and test.zip to data/raw directory. If the Kaggle API is installed, run following command.

```bash
Use the Kaggle API to download the dataset.
https://github.com/Kaggle/kaggle-api

$ kaggle competitions download -c cs-t0828-2020-hw1
$ mkdir -p datasets
$ unzip cs-t0828-2020-hw1.zip -d ./datasets
```

## Training


My final submission is use resnet50,and use the k-folder to vote for the final category of images.

Run `train_index.py` to train.

Train with last stage and 3 positive images (ResNet-50 7x7):
```bash
python train_index.py --stage 3 --num_positive 3
```

The expected training times are:

Model | GPUs | Image size | Training Epochs | Training Time
------------ | ------------- | ------------- | ------------- | -------------
resnet50 | 2x 2080Ti | 512 | 180 | 21 hours
## Pretrained models
You can download pretrained model that used for my submission from [link](https://drive.google.com/file/d/1-x33sKGFGgp22Tg7Z_0B6QzDwEcAdp1R/view). 


Unzip them into results then you can see following structure:
```bash
pretrain_model
  +- 
  |  +- 0.pth
  |  +- 1.pth
  |  +- 2.pth
  |  +- 3.pth
  |  +- 4.pth
  |  +- 5.pth
  |  +- 6.pth
```

##Inference
If trained weights are prepared, you can create files that contains class of images.
```bash
$ python test_index.py --resume={trained model path} 
```

##Make Submission
Following command will ensemble of all models and generate submission file (result.csv) at root.
```bash
$ python make_submission.py \
 --path {The base abspath of predicted result (~/netmoddl)} \
 --k-folder {True if using k-folder}
```

## Citation
```bib
@InProceedings{Zhou_2020_CVPR,
author = {Zhou, Mohan and Bai, Yalong and Zhang, Wei and Zhao, Tiejun and Mei, Tao},
title = {Look-Into-Object: Self-Supervised Structure Modeling for Object Recognition},
booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2020}
}
```

