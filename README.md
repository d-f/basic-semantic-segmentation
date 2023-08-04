# basic-semantic-segmentation

Speeding Up IoU and Dice measurement
---------------------------------------------------------------------------------------------------------------------------------------------------------------------

Measuring IoU and Dice on the test set was taking an inordinate amount of time (~5 hours to evaluate the entire test set). It was found that the one hot encoding step was taking a the longest amount of time. The original function iterated through each pixel in the ground truth and prediction arrays, isolated all the values for each channel at this specific pixel location, determined the largest class, one hot encoded the channel values and updated the array. This was sped up by determining the max value for each channel in one step with torch.argmax(dim=0) and the one hot encoded array was created with torch.nn.functional.one_hot() (see onehot_3d_array() on line 246 in utils/data_utils.py). This sped the time it takes to evaluate the test set from 5 hours to less than a minute with GPU acceleration.

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
- Resume training on a specific epoch: (load previous model / optimizer parameters, append additional training data to train result file and re-create test file)
```
python develop_resnet_FCN.py --pretrained -num_classes 37 -batch_size 20 -patience 5 -result_dir "C:\\ml_projects\\fcn_segmentation\\results\\" -train_result_filename "resnet101_fcn_1_train_results.json" -test_result_filename "resnet101_fcn_1_test_results.json" -lr 1e-4 -model_save_name resnet101_fcn_1.pth.tar -num_epochs 64 -data_root "C:\\ml_projects\\fcn_segmentation\\pcam_data\\" --continue_bool -start_epoch 32
```

Fully convolutional network background information
---------------------------------------------------------------------------------------------------------------------------------------------------------------------

Fully convolutional networks are made up entirely of convolutional layers. They generally follow an encoder-decoder structure where the encoder first distills an image down to a set of feature maps similar to the first section of convolutional neural networks with classifiers. The decoder makes use of deconvolutions (or transpose convolutions) to turn the set of feature maps into a larger segmentation map. The stride length of a convolution determines the amount the filter or kernel moves across the input at a time. Transpose convolutions differ from convolutions in that their stride length is less than one, resulting in a larger output than input.
