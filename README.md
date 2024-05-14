# basic-semantic-segmentation
In order to run:
---------------------------------------------------------------------------------------------------------------------------------------------------------------------
- Set up environment 
  - Download Anaconda [https://www.anaconda.com/download](https://www.anaconda.com/download)
  - Create a conda environment (in this case named segmentation_env)
```
conda create -n segmentation_env python=3
```
  - Clone this repository to any directory, in this case C:\\ml_code\\basic-semantic-segmentation\\
```
cd C:\ml_code\
git clone https://github.com/d-f/basic-semantic-segmentation.git
```
  - Download dependencies
```
cd basic-semantic-segmantation
pip install -r requirements.txt --find-links https://download.pytorch.org/whl/torch_stable.html
```
  - Set up directories
    - Depending on OS, run either create_dirs.ps1 (for Windows) or create_dirs.sh (for Linux) and choose a "project directory" for everything to be added to, in this case "C:\\ml_projects\\fcn_segmentation\\"
```
C:\ml_code\basic-semantic-segmentation\create_dirs.ps1 "C:\\ml_projects\\fcn_segmentation\\"
```
or  
```
bash create_dirs.sh
"/C/ml_projects/fcn_segmentation/"
```
- Train a model
 ```
python develop_resnet_FCN.py --pretrained -num_classes 37 -batch_size 20 -patience 5 -result_dir "C:\\ml_projects\\fcn_segmentation\\results\\" -train_result_filename "resnet101_fcn_1_train_results.json" -test_result_filename "resnet101_fcn_1_test_results.json" -lr 1e-4 -model_save_name resnet101_fcn_1.pth.tar -num_epochs 64 -data_root "C:\\ml_projects\\fcn_segmentation\\pcam_data\\"
```
- Resume training on a specific epoch: (load previous model, append additional training data to train result file and re-create test file)
```
python develop_resnet_FCN.py --pretrained -num_classes 37 -batch_size 20 -patience 5 -result_dir "C:\\ml_projects\\fcn_segmentation\\results\\" -train_result_filename "resnet101_fcn_1_train_results.json" -test_result_filename "resnet101_fcn_1_test_results.json" -lr 1e-4 -model_save_name resnet101_fcn_1.pth.tar -num_epochs 64 -data_root "C:\\ml_projects\\fcn_segmentation\\pcam_data\\" --continue_bool -start_epoch 32
```
