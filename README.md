# basic-semantic-segmentation

In order to re-create the results presented:

Install [detectron2](https://github.com/facebookresearch/detectron2) on Windows:
```
conda create -n detectron2_env python=3
conda activate detectron2_env
git clone https://github.com/facebookresearch/detectron2.git
python -m pip install -e detectron2
```

Download and extract the [kaggle dataset](https://www.kaggle.com/datasets/andrewmvd/liver-tumor-segmentation) to a directory, in this case C:\liver_segmentation\

This dataset was extracted from LiTS – Liver Tumor Segmentation Challenge (LiTS17) organised in conjunction with ISBI 2017 and MICCAI 2017.

The dataset contains .nii files of 130 contrast-enhanced abdominal CT scans with segmentation masks for liver and tumor tissue.

