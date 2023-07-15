# basic-semantic-segmentation
- how to run
- results of best performing model
- theory behind fully convolutional networks

Fully convolutional networks, as the name suggests only make use of convolutions. Other segmentation networks like Mask-RCNN work by applying a CNN with a bounding box output to predict bounding boxes, the boxes are combined together and a fully convolutional network predicts the mask within the bounding box **check**. Fully convolutional networks can directly predict the mask by making use of deconvolutions (or transpose convolutions) to convert the feature maps to segmentation maps rather than bounding boxes. The transpose convolution works similarly to a normal convolution, where a small filter with a stride greater than one is convolved across an image to produce a smaller feature map but instead transpose convolutions make use of a fractional stride, resulting in a larger feature map than the original input. 
