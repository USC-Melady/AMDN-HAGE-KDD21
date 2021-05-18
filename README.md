# Identifying Coordinated Accounts on Social Media through Hidden Influence and Group Behaviours

## AMDN-HAGE (Pytorch implementation)
Coordinated Activity Detection, Disinformation/Influence Campaigns, Social Media,Fake News, Temporal Point Process

## Usage
AMDN-HAGE code is built for Coordinated Activity Detection and is based on Neural Temporal Points Processes (NTPP) and Gaussian Mixture Models (some code functionality of NTPP is built on top of existing repository-https://github.com/shchur/ifl-tpp). 

AMDN-HAGE repository is self-contained and can be run as a standalone repository. In order to run the code (Example script: code/run_cmd/run.bash), you need to first install the `dpp` library:
```bash
cd code
python setup.py install
```

## Requirements
```
numpy=1.16.4
pytorch=1.4.0
scikit-learn=0.24.0
scipy=1.6.0
```
Currently the code is hardcoded using pytorch DataParallel to utilize 4-GPU machines to speed up computations for very large social network datasets. Please modify it as required for your system requirements.


## Running the code
For example, for activity traces (sequences) of social media, we can identify coordinated accounts by first using AMDN-HAGE to output the learned account embeddings.
```Example script: code/run_cmd/run.bash```
Then clustering these embeddings to obtain the coordinated (anomalous) and normal groups.
```code/clustering.ipynb```
Dataset for COVID-19 used in KDD'21 can be accessed here:
``` https://drive.google.com/drive/folders/1PrG50ZJdRP7-tmUMj2sqcMfQ1NMCTKVN?usp=sharing ```


## Please cite the following work.
```
@inproceedings{,
author = {Sharma, Karishma and Zhang, Yizhou and Ferrara, Emilio and Liu, Yan},
title = {Identifying Coordinated Accounts on Social Media through Hidden Influence and Group Behaviours},
year = {2021},
publisher = {Association for Computing Machinery},
booktitle = {Proceedings of the 27th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining},
series = {KDD'21}
}
```