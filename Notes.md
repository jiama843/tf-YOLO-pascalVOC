# YOLO (You Only Look Once)

God Tier explanations: 
http://machinethink.net/blog/object-detection-with-yolo/

https://www.youtube.com/watch?v=NM6lrxy0bxs&


https://youtu.be/NM6lrxy0bxs

https://www.youtube.com/watch?v=gKreZOUi-O0&


https://youtu.be/gKreZOUi-O0

Description
Convert object detection into a regression problem with spatially separated boxes and associating class probabilities.

We use a single neural network and determine boxes, then class probabilities in one evaluation. (Fast because one run evaluates an image).


![](https://d2mxuefqeaa7sj.cloudfront.net/s_66214D52DBE7F36AF1C66554D9677BC10BCAE7F857A47B855008AC989476B083_1534392487671_image.png)

![](https://d2mxuefqeaa7sj.cloudfront.net/s_66214D52DBE7F36AF1C66554D9677BC10BCAE7F857A47B855008AC989476B083_1534392506687_image.png)


General Overview
An input image is divided into an S by S grid. 

Each grid predicts:


1. B bounding boxes and their respective confidence scores

Each bounding box contains 5 predictions:

- $$x$$ → $$x$$ coordinate of the center position of the boundary box (relative to grid cell)
- $$y$$ → $$y$$ coordinate of the center position of the boundary box (relative to grid cell)

$$(x,y)$$ represents the center of the boundary box


- $$w$$ → width relative to the full image (percentage of grid cell taken)
- $$h$$ → height relative to the full image (percentage of grid cell taken)
- Confidence → (For the sake of simplicity) The probability that it belongs to a certain class. It is defined as Pr(Object) ∗ (intersection over union of (expected and prediction).


   $$Confidence = Pr(Object)*\frac{expected\ \cap\ prediction}{expected\ \cup\ prediction}$$


2. C conditional class probabilities 

This is defined as the probability that the box contains some class (class_i) given that it has an object.


   $$Probability = Pr(Class_i|Object)$$

In the end, the conditional class probability and individual box confidence predictions are multiplied together to get class specific confidence scores for each box.

The result encodes both the probability of a class (class_i) appearing in the box and how well it fits the object.

$$Confidence * Probability=Pr(Class_i)*\frac{expected\ \cap\ prediction}{expected\ \cup\ prediction}$$


![](https://d2mxuefqeaa7sj.cloudfront.net/s_66214D52DBE7F36AF1C66554D9677BC10BCAE7F857A47B855008AC989476B083_1534396249623_image.png)

![Note: the 5 is due to the 5 predictions associated with each boundary box](https://d2mxuefqeaa7sj.cloudfront.net/s_66214D52DBE7F36AF1C66554D9677BC10BCAE7F857A47B855008AC989476B083_1534616806366_image.png)


* Reference for confidence: http://hunch.net/?p=317

# The Neural Network Model

![The YOLO model](https://i.stack.imgur.com/bsnvR.png)


Analysis of the first convolutional layer → The kernel has a size of 7x7. There are 64 convolution filters in the first layer. The stride is 2.

Analysis of the first Maxpool layer → The kernel has a size of 2x2. The stride is 2.


![YOLO network analysis](https://image.slidesharecdn.com/yolo-170616085751/95/pr12-you-only-look-once-yolo-unified-realtime-object-detection-8-1024.jpg?cb=1497603506)


Training the Network


![YOLO network analysis (Training)](https://image.slidesharecdn.com/yolo-170616085751/95/pr12-you-only-look-once-yolo-unified-realtime-object-detection-11-638.jpg?cb=1497603506)


The result of the network model is a S x S x (B * 5 + C) = 7x7x30 tensor (Using VOC) that contains class probability predictions and boundary box predictions. 

In the case of the Object detection track dataset, C=500, B=4, S=14, so the result is a 14x14x520, tensor.

Optimizations

![YOLOv2](https://d2mxuefqeaa7sj.cloudfront.net/s_66214D52DBE7F36AF1C66554D9677BC10BCAE7F857A47B855008AC989476B083_1534479481546_image.png)

![YOLOv3](https://d2mxuefqeaa7sj.cloudfront.net/s_66214D52DBE7F36AF1C66554D9677BC10BCAE7F857A47B855008AC989476B083_1534479684816_image.png)

