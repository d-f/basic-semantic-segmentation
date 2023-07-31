# basic-semantic-segmentation
How speed of measuring segmentation performance was improved:
Measuring IoU was taking an inordinate amount of time (~5 hours) by using the equation:

![image](https://github.com/d-f/basic-semantic-segmentation/assets/118086192/e1dcfc95-1c94-4e78-8d1d-c97067ec4bcc)

The functions were sped up by changing to the equation:
 
 IoU = TP / (TP + FN + FP)

 Switching from one-hot encoding functions that iterated through channel values for each pixel, determined the predicted class, one-hot encoded the channel values and updated the array to built-in torch.argmax and torch.nn.functional.one_hot methods. Now testing the model and measuring IoU and Dice for each prediction takes less than a minute.

 In order to run:

- download pet dataset
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
