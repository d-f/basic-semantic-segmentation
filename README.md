# basic-semantic-segmentation

In order to re-create the results presented:

Install [detectron2](https://github.com/facebookresearch/detectron2) on Windows:
```
conda create -n detectron2_env python=3
conda activate detectron2_env
git clone https://github.com/facebookresearch/detectron2.git
python -m pip install -e detectron2
```

Download and extract the [kaggle dataset](https://www.kaggle.com/datasets/andrewmvd/breast-cancer-cell-segmentation) to a directory, in this case C:\

This dataset contains histopathology 58 images of breast cancer tissue stained with Hematoxylin and Eosin (H&E) with tumor cells segmented.
