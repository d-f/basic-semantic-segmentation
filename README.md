# basic-semantic-segmentation
---------------------------------------------------------------------------------------------------------------------------------------------------------------------
Speeding Up IoU and Dice measurement
---------------------------------------------------------------------------------------------------------------------------------------------------------------------

Measuring IoU and Dice was taking an inordinate amount of time (~5 hours to evaluate the entire test set). It was found that the one hot encoding step was taking a the longest amount of time. The original function iterated through each pixel in the ground truth and prediction arrays, isolated all the values for each channel at this specific pixel location, determined the largest class, one hot encoded the channel values and updated the array. This was sped up by determining the max value for each channel in one step with torch.argmax(dim=0) and the one hot array was created with torch.nn.functional.one_hot(). This sped the time it takes to evaluate the test set from 5 hours to less than a minute with GPU acceleration.

 In order to run:

- Download Anaconda [https://www.anaconda.com/download](https://www.anaconda.com/download)
- Create a conda environment (in this case named segmentation_env)
```
conda create -n segmentation_env python=3
```
- set up directories
 
 ```
bash develop_resnet_FCN.py --pretrained -num_classes 37 -batch_size 20 -patience 5 -result_dir "" -train_result_filename "" -test_result_filename ""
-lr 1e-4 -model_save_name resnet101_fcn_1.pth.tar -num_epochs 64 -data_root ""
```
To resume training on a specific epoch:
```
bash develop_resnet_FCN.py --pretrained -num_classes 37 -batch_size 20 -patience 5 -result_dir "" -train_result_filename "" -test_result_filename ""
-lr 1e-4 -model_save_name resnet101_fcn_1.pth.tar -num_epochs 64 -data_root "" --continue_bool -start_epoch 32
```

**Fully connected network background information**

Fully convolutional networks are made up entirely of convolutional layers. They generally follow an encoder-decoder structure where the encoder first distills an image down to a set of feature maps similar to the first section of convolutional neural networks with classifiers. The decoder makes use of deconvolutions (or transpose convolutions) to turn the set of feature maps into a larger segmentation map. The stride length of a convolution determines the amount the filter or kernel moves across the input at a time. Transpose convolutions differ from convolutions in that their stride length is less than one, resulting in a larger output than input.
