# AstMatch: Adversarial Self-training Consistency Framework for Semi-Supervised Medical Image Segmentation
## Introduction
Official code for "[AstMatch: Adversarial Self-training Consistency Framework for Semi-Supervised Medical Image Segmentation](https://arxiv.org/abs/2406.19649)".
## Requirements
This repository is based on PyTorch 2.0.1, CUDA 11.8 and Python 3.10.11. All experiments in our paper were conducted on NVIDIA GeForce RTX 4090 GPU.
## Dataset
Data could be got at [LA](https://github.com/yulequan/UA-MT/tree/master/data), [ACDC](https://github.com/HiLab-git/SSL4MIS/tree/master/data/ACDC), and [Pancreas-NIH](https://wiki.cancerimagingarchive.net/display/Public/Pancreas-CT).

To preprocess the Pancreas-NIH dataset, please use the `preprocess/pancreas_preprocess.py` script. This script will convert the raw data into HDF5 format (`.h5` files) for easier handling and training.

The `preprocess/pancreas_preprocess.py` script is based on the implementation from the [CoraNet repository](https://github.com/koncle/CoraNet).
## Dataset Splits
We provide the dataset splits in the `data_split` folder.

For the Pancreas-NIH dataset, we follow the data split strategy used in the [CoraNet repository](https://github.com/koncle/CoraNet):
1. **When the number of labeled samples is 12**:
   - Labeled data: `0001.h5` to `0012.h5`.
   - Unlabeled data: The remaining training data.

2. **When the number of labeled samples is 6**:
   - Labeled data: `0001.h5` to `0006.h5`.
   - Unlabeled data: The remaining training data.
## Usage
We provide the training and testing code in the `code` folder and the trained models in the `models` folder.

To train a model,
```
python ./code/LA_AstMatch_train.py  #for LA training
python ./code/ACDC_AstMatch_train.py  #for ACDC training
python ./code/Pancreas_AstMatch_train.py  #for Pancreas-NIH training
``` 

To test a model,
```
python ./code/test_LA.py  #for LA testing
python ./code/test_ACDC.py  #for ACDC testing
python ./code/test_Pancreas.py  #for Pancreas-NIH testing
```

## Citation

If our AstMatch is useful for your research, please consider citing:

```bibtex
@article{zhu2024astmatch,
  title={AstMatch: Adversarial Self-training Consistency Framework for Semi-Supervised Medical Image Segmentation},
  author={Zhu, Guanghao and Zhang, Jing and Liu, Juanxiu and Du, Xiaohui and Hao, Ruqian and Liu, Yong and Liu, Lin},
  journal={arXiv preprint arXiv:2406.19649},
  year={2024}
}
```

## Acknowledgements
Our code is largely based on [BCP](https://github.com/DeepMed-Lab-ECNU/BCP/tree/main) and [SSL4MIS](https://github.com/HiLab-git/SSL4MIS). We appreciate the valuable work of these authors and hope that our work can also contribute to the research on semi-supervised medical image segmentation.

## Questions
If you have any questions, welcome contact me at [gzhu663663@gmail.com](mailto:gzhu663663@gmail.com).
