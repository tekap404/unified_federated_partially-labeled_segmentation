# unified_federated_partially-labeled_segmentation

## Introduction

UFPS is an effective federated learning framework for partially annotated datasets. It consists of ULL for class heterogeneity and sUSAM for client drift.

## Citation

*Jiang, L., Ma, L.Y., Zeng, T.Y., & Ying, S.H. (2024). UFPS: A unified framework for partially annotated federated segmentation in heterogeneous data distribution. Patterns.*

```
@article{Jiang2024UFPSAU,
  title={UFPS: A unified framework for partially annotated federated segmentation in heterogeneous data distribution},
  author={Le Jiang and Li Yan Ma and Tie Yong Zeng and Shi Hui Ying},
  journal={Patterns},
  year={2024},
  url={https://api.semanticscholar.org/CorpusID:267277291}
}
```

## Abstract
Partially supervised segmentation is a label-saving method based on datasets with fractional classes labeled and intersectant. However, its practical application in real-world medical scenarios is hindered by privacy concerns and data heterogeneity. To address these issues without compromising privacy, federated partially supervised segmentation (FPSS) is formulated in this work. The primary challenges for FPSS are class heterogeneity and client drift. We propose a Unified Federated Partially-labeled Segmentation (UFPS) framework to segment pixels within all classes for partially-annotated datasets by training a comprehensive global model which avoids class collision. Our framework includes Unified Label Learning and sparsed Unified Sharpness Aware Minimization for unification of class and feature space, respectively. Through empirical studies, we found that traditional methods in partially supervised segmentation and federated learning often struggle with class collision when combined. Our extensive experiments on real medical datasets demonstrate better deconflicting and generalization capabilities of UFPS. 

## Environment

```
conda create -n UFPS python=3.8 -y
conda activate UFPS
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
pip install -r requirements.txt
```

## Dataset & Pre-processing

### Download datasets

- Download [WORD](https://github.com/HiLab-git/WORD) and put the 'WORD-V0.1.0' folder under 'data/WORD'. Move or copy all files in 'imagesVal' folder to 'imagesTr' folder. Move or copy all files in 'labelsVal' folder to 'labelsTr' folder.

- Download [AbdomenCT-1K](https://github.com/JunMa11/AbdomenCT-1K). Creat a folder named 'AbdomenCT' and another 'imagesTr' folder under 'AbdomenCT'. Move all files in 'AbdomenCT-1K-ImagePart1', 'AbdomenCT-1K-ImagePart2', 'AbdomenCT-1K-ImagePart3' folders into the 'AbdomenCT/imagesTr' folder. Rename 'Mask' folder as 'labelsTr' and put it under the 'AbdomenCT' folder. Put the 'AbdomenCT' folder under 'data/AbdomenCT-1K'.

- Download [AMOS2022](https://amos22.grand-challenge.org/) and put the 'AMOS22' folder under 'data/AMOS2022'.

- Download [BTCV](https://www.synapse.org/#!Synapse:syn3193805/wiki/217752) (Join the challenge to see hidden files.). Create new folders 'imagesTr' and 'labelsTr' under 'data/BTCV'. Put all files in 'Abdomen/RawData/Training/img' and 'Abdomen/RawData/Training/label' folders under 'data/BTCV/imagesTr' and 'data/BTCV/labelsTr', respectively.

  ```
  UFPS-main
  ┣ ...
  ┣ data
  ┣ ┣ WORD
  ┣ ┣ ┣ WORD-V0.1.0
  ┣ ┣ ┣ ┣ imagesTr
  ┣ ┣ ┣ ┣ ┣ word_0001.nii.gz
  ┣ ┣ ┣ ┣ ┗ ...
  ┣ ┣ ┣ ┣ labelsTr
  ┣ ┣ ┗ ...
  ┣ ┣ AbdomenCT-1K
  ┣ ┣ ┣ AbdomenCT
  ┣ ┣ ┣ ┣ imagesTr
  ┣ ┣ ┣ ┣ ┣ Case_00001_0000.nii.gz
  ┣ ┣ ┣ ┣ ┣ ...
  ┣ ┣ ┣ ┣ labelsTr
  ┣ ┣ ┗ ...
  ┣ ┣ AMOS2022
  ┣ ┣ ┣ AMOS22
  ┣ ┣ ┣ ┣ imagesTr
  ┣ ┣ ┣ ┣ ┣ amos_0001.nii.gz
  ┣ ┣ ┣ ┣ ┗ ...
  ┣ ┣ ┣ ┣ labelsTr
  ┣ ┣ ┗ ...
  ┣ ┣ BTCV
  ┣ ┣ ┣ BTCV
  ┣ ┣ ┣ ┣ imagesTr
  ┣ ┣ ┣ ┣ ┣ img0001.nii.gz
  ┣ ┣ ┣ ┣ ┗ ...
  ┣ ┣ ┣ ┣ labelsTr
  ┣ ┣ ┗ ...
  ┗ ...
  ```

### Pre-processing

All 'your_path' should be replaced to your path saving the 'UFPS-main' folder.

1. Per dataset processing. Let's take WORD for an example:
   - cd /your_path/data/WORD
   - sh preprocess.sh
   - To pre-processing other datasets, replace 'WORD' in the cd command to 'AbdomenCT-1K' / 'AMOS2022' / 'BTCV' and use the sh command.
2. Get all file paths.
   - cd /your_path/data/client_data
   - sh get_all_filepath.sh

## Training

Note: 

- Most configurations are in '/your_path/code/configs/setting1_config.py'
- If you start multiple train codes at the same time, remember to change the number in 'cfg.dist_url' after ''tcp://127.0.0.1:'.

### Pretrain teacher models

```
cd /your_path/code
```

Single-GPU:

```
CUDA_VISIBLE_DEVICES=0 python train_solo_main.py --multi_gpu 0 --gpu_num 1 --exp_folder solo_client1_part123 --client_name client_1
```

Multi-GPU:

```
CUDA_VISIBLE_DEVICES=1,2 python train_solo_main.py --use_multi_gpu True --multi_gpu 0,1 --gpu_num 2 --exp_folder solo_client1_part123 --client_name client_1
```

### UFPS training

```
cd /your_path/code
```

Single-GPU:

```
CUDA_VISIBLE_DEVICES=0 python train_main.py --multi_gpu 0 --gpu_num 1 --exp_folder UFPS
```

Multi-GPU:

```
CUDA_VISIBLE_DEVICES=1,2 python train_main.py --use_multi_gpu True --multi_gpu 0,1 --gpu_num 2 --exp_folder UFPS
```

## Evaluation

SOLO:

```
cd /your_path/code
CUDA_VISIBLE_DEVICES=0 python test_main.py --multi_gpu 0 --gpu_num 1 --plot False --test_mode SOLO --train_site client_1
```

Change 'client_1' after '--train_site' if you want to train on other clients. 

Always remember to change the path of 'cfg.test_path' in '/your_path/code/configs/setting1_config.py' !

FL:

```
cd /your_path/code
CUDA_VISIBLE_DEVICES=0 python test_main.py --multi_gpu 0 --gpu_num 1 --plot False --test_mode FL --train_site all
```

Always remember to change the path of 'cfg.test_path' in '/your_path/code/configs/setting1_config.py' !

## Pretrained weights

Pretrained checkpoints and data division files can be found at [google drive](https://drive.google.com/drive/folders/14kG1tZbLTtmV7zzWZ0QzDLdVxMI-vK3E?usp=drive_link).
