# Data-Analysis-20202
Final project
### Directory Tree
```
ROOT_PATH
├── data
├── dataloader
│   └── dataset.py
├── main.py
├── README.md
├── requirements.txt
└── utils
    └── log.py
```
## Preperation

### Dependencies
This repository require Python 3.8+.
First install Pytorch 1.8.1 from [Pytorch.org](https://pytorch.org) 
Install all python's dependencies by running:
```
pip install -r requirements.txt
```

### Clustering Dataset
The raw dataset must have the following format:
```
ROOT_PATH
├── CLASS_1
│   ├── image1.ext
│   ├── image2.ext
│   ├── ...
│   └── imageN.ext    
├── CLASS_2
│   ├── image1.ext
│   ├── image2.ext
│   ├── ...
│   └── imageN.ext    
...
├── CLASS_N
│   ├── image1.ext
│   ├── image2.ext
│   ├── ...
│   └── imageN.ext    
```
All the class names must be defined at `dataloader/classes.yml`

## Usage Commands:
```
usage: main.py [-h] [--data_path DATA_PATH] [--exp_name EXP_NAME] [--min_cluster MIN_CLUSTER] [--max_cluster MAX_CLUSTER]
               [--feature_extractor FEATURE_EXTRACTOR]

optional arguments:
  -h, --help            show this help message and exit
  --data_path DATA_PATH
                        Path to data directory
  --exp_name EXP_NAME   Set experiment directory name
  --min_cluster MIN_CLUSTER
                        Min number of clusters
  --max_cluster MAX_CLUSTER
                        Max number of clusters
  --feature_extractor FEATURE_EXTRACTOR
                        Name of Feature extractor. Remeber to comment GrayScale transform in dataset
```