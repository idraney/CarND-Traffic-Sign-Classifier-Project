# **Traffic Sign Recognition** 

## Writeup (README.md)

---

**Build a Traffic Sign Recognition Project**

The goals of this project are the following:
* Load the training, validation, and testing data sets
* Explore, summarize, and visualize the training data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

[//]: # (Image References)

[image1]: ./writeupimages/visualization.png "Visualization: Histogram of the initial training data set."
[image2]: ./writeupimages/normalize_before_after.png "Images before and after normalization."
[image3]: ./writeupimages/gaussian_noise_before_after.png "Images before and after Gaussian noise addition (brighten and darken)"
[image4]: ./writeupimages/rotation_before_after.png "Images before and after rotation at -10, -5, +5, and +10 degrees"
[image5]: ./writeupimages/perspective_transform_before_after.png "Images before and after perspective transformations"
[image6]: ./writeupimages/grayscale_before_after.png "Images before and after grayscale conversion"
[image7]: ./writeupimages/grayscale_before_after.png "Images before and after being scaled up 15%"
[image8]: ./writeupimages/visualization_after.png "Visualization: Histogram of the augmented training data set"
[image9]: ./writeupimages/internet_images.png "30 traffic sign images found on the web, labeled"
[image10]: ./writeupimages/internet_images_predictions.png "30 traffic sign images found on the web, labeled, predicted"
[image11]: ./writeupimages/internet_images_softmax.png "Top 5 softmax probabilities of the 30 traffic sign images found on the web"
[image12]: ./writeupimages/activation_layers.png "Visualization of the activation layers"



## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. 

You're reading it! Here is a link to this [project's code](./Traffic_Sign_Classifier.ipynb).  An HTML version of the project is linked [here](./Traffic_Sign_Classifier.html).

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

Python and numpy were used to calculate summary statistics of the traffic signs data set:

* The size of the initial training set (before augmention and training) is 34799 images
* The size of the final training set (after augmention and training) is 3948200 images
* The size of the validation set is 4410 images
* The size of test set is 12630 images
* The shape of a traffic sign image is 32 pixels by 32 pixels (RGB 3-channel image)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the initial training data set. It is a bar chart showing the number of images per label ID.

![alt text][image1]



### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

Being that the initial training data set contained an image count ranging from 180 to 2010 images per label.  The data set needed to be augmented to increase the number of images, particularly for those labels with a low count.  The resulting images from each augmentation step were added to the training set to increase the number of overall images.  Due to limited memory on the local machine, each image class (i.e., label ID) was restricted to a maximum count of 90000 images. 

The first augmentation step was to normalize the images.  The normalize augmentation increased the training dataset to a total of 69598 images.  The figure below is an example of three images from the training data set before and after normalization.

![alt text][image2]

The second augmentation step was to add gaussian noise to the images.  This step was done in two parts: 1) darken images and 2) brighten images.  The gaussian noise augmentation increased the training dataset to a total of 208794 images.  The figure below is an example of two images from the training data set before and after gaussian noise augmentation (darken and brighten).

![alt text][image3]

The third augmentation step was to rotate the images at small angles.  The angles which the images were rotated were -10, -5, +5, and +10 degrees.  The rotation augmentation increased the training dataset to a total of 1043970 images.  The figure below is an example one image from the training data set before and after each of the rotations.

![alt text][image4]

The fourth augmentation step was to perform perspective transforms on the training data set.  The perspective transformations performed were  top, bottom, left, and right perspectives from the original images.  Image labels were limited to a maximum count of 50000 for the perspective transform augmentation. The perspective transform augmentation increased the training dataset to a total of 2644250 images.   The figure below is an example one image from the training data set before and after each of the perspective transformations.

![alt text][image5]

The fifth augmentation step was to create grayscale images from the existing data set.  Image labels were limited to a maximum count of 50000 for the grayscale conversion.  The grayscale augmentation increased the training dataset to a total of 2845250 images.   The figure below is an example one image from the training data set before and after each of the perspective transformations.

![alt text][image6]

The sixth augmentation step was to scale images up 15% from the existing data set.  The scaling was confined within the image size of 32 pixels by 32 pixels.  Image labels were limited to a maximum count of 90000 for scaling.  The scale augmentation increased the training dataset to a total of 3948200 images.   The figure below is an example one image from the training data set before and after each of the perspective transformations.

![alt text][image7]

The scale augmentation was the final augmentation step for the training data set.  The resulting set of 3948200 total images can be seen in the histogram in the figure below, separated by label ID.

![alt text][image8]



#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The model architecture was based on the LeNet-5 neural network architecture.  Changes made were the mean (mu) to 0.05, and adding a dropout layer after the max pooling layer after the second convolutional layer.  Other modifications to the LeNet-5 model were attempted, but later abandoned, as they did not make any significant improvement to the training results.  These attempted modifications can be seen in the [source code](./Traffic_Sign_Classifier.ipynb).  The table below shows the complete model architecture.

| Layer					| Description														| 
|:---------------------:|:-----------------------------------------------------------------:| 
| Input					| 32x32x3 RGB image   												| 
| Convolution 3x3		| 1x1 stride, same padding, outputs 28x28x6 						|
| RELU					| Activation														|
| Max pooling			| 2x2 stride, outputs 14x14x6  										|
| Convolution 3x3		| 1x1 stride, same padding, outputs 10x10x16 						|
| RELU					| Activation														|
| Max pooling	      	| 2x2 stride, outputs 5x5x16  										|
| Dropout				| Keep probability set to 1 during evaluation, 0.5 in training 		|
| Flatten 				| Outputs 400 														|
| Fully Connected		| Outputs 120 														|
| RELU					| Activation														|
| Fully Connected		| Outputs 84 														|
| RELU					| Activation														|
| Fully Connected		| Outputs 43 \[logits\] (i.e., the number of image classes) 		|

 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

In the neural network training pipeline, the logits returned from the LeNet-5 architecture were used to measure the probability error with one-hot encoding of the 43 image classes using `tf.nn.softmax_cross_entropy_with_logits`.  The mathematical mean of the probability error (cross entropy) across the tensor was then computed using `tf.nn.reduce_mean`.  the Adam algorithm was used for optimization using `tf.train.AdamOptimizer` with a learning rate of 0.001.  The training operation was gathered bycomputing the gradients and applying them to the variables from the optimized output and the output of the probability error.

During model evaluation, `tf_equal` was used to return the element-wise values of the logits and one-hot encoded image labels.  `tf_reduce_mean` was then used to calculate the model accuracy.  Finally, the model was saved to later be applied to the validation and test data sets, as well as additional images found on the internet.

Training with the data set iterated over 20 epochs, each with a batch size of 512.



#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

Model accuracy could vary slightly on each of the data sets on each code run.  The model results of the final run were:
* Training set accuracy: 		99.8%
* Validation set accuracy of: 	96.7%
* Test set accuracy of: 		95.0%

An iterative approach was chosen to obtain the highest model accuracy possible:
* The first architecture that was tried was the LeNet-5 architecture.  It was chosen because it was taught in the Udacity Self-Driving Car Engineer Nanodegree program, and was shown to be effective with the MNIST data set.
* The initial architecture showed problems in its default state that produced a model accuracy lower than what was acceptable (93%).
* As previously described, enhancements to the model were made, such as adding a dropout layer after the second max pooling layer.  The high accuracy of the training, validation, and testing data sets indicate that there was no underfitting or overfitting in the final model.
* Changing the standard deviation hyperparameter (sigma) showed no improvement to the model, so it was finally left at its default value of 0.0.  The mean hyperparameter (mu) had a default value of 0.1, but the model showed significant improvement after it was changed to 0.05.
* Which parameters were tuned? How were they adjusted and why?
* Convolutional layers work exceptionally well on image data.  Hence, the LeNet-5 model architecture was well suited for the task of traffic sign image classification because of its convolutional layers.  The addition of a dropout layer improved the model by randomly dropping nodes and preventing overfitting.



### Test a Model on New Images

#### 1. Choose thirty German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Thirty German traffic sign images were found on the web.  Each image was labeled using the classification list [signnames.csv](./signnames.csv).  The figure below shows the images that were used.

![alt text][image9]

The collection of images are a mixture of photographs of traffic signs and pictograph representations of traffic signs.  It was unknown how the model would perform on each individual image.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the model prediction for each image found on the web:

| Image												| Prediction										| 
|:-------------------------------------------------:|:-------------------------------------------------:| 
| 21,Double curve 									| 21,Double curve 									| 
| 12,Priority road 									| 12,Priority road 									| 
| 17,No entry 										| 17,No entry 										| 
| 4,Speed limit (70km/h) 							| 4,Speed limit (70km/h) 							| 
| 12,Priority road 									| 12,Priority road 									| 
| 25,Road work 										| 25,Road work 										| 
| 33,Turn right ahead 								| 33,Turn right ahead 								| 
| 13,Yield 											| 13,Yield 											| 
| 5,Speed limit (80km/h) 							| 1,Speed limit (30km/h) 							| 
| 25,Road work 										| 25,Road work 										| 
| 1,Speed limit (30km/h) 							| 0,Speed limit (20km/h) 							| 
| 26,Traffic signals 								| 26,Traffic signals 								| 
| 2,Speed limit (50km/h) 							| 2,Speed limit (50km/h) 							| 
| 19,Dangerous curve to the left 					| 19,Dangerous curve to the left 					| 
| 23,Slippery road 									| 23,Slippery road 									| 
| 40,Roundabout mandatory 							| 40,Roundabout mandatory 							| 
| 38,Keep right 									| 33,Turn right ahead 								| 
| 31,Wild animals crossing 							| 31,Wild animals crossing 							| 
| 15,No vehicles 									| 15,No vehicles 									| 
| 9,No passing 										| 9,No passing 										| 
| 10,No passing for vehicles over 3.5 metric tons 	| 10,No passing for vehicles over 3.5 metric tons 	| 
| 1,Speed limit (30km/h) 							| 1,Speed limit (30km/h) 							| 
| 37,Go straight or left 							| 21,Double curve 									| 
| 18,General caution 								| 18,General caution 								| 
| 28,Children crossing 								| 28,Children crossing 								| 
| 14,Stop 											| 14,Stop 											| 
| 17,No entry 										| 17,No entry 										| 
| 26,Traffic signals 								| 26,Traffic signals 								| 
| 27,Pedestrians 									| 18,General caution 								| 
| 36,Go straight or right 							| 36,Go straight or right 							| 


The model was able to correctly predict 26 of the 30 traffic signs, resulting in 86.7% accuracy in the final run.  Previous runs had even resulted in higher accuracy.  This compares favorably to the accuracy on the test set of 95.0%.

There appears to be no significant difference between the model performance on traffic sign photographs and pictograph representations of traffic signs.  Of the total 4 prediction failures, 2 failures were of photographs (50%), and 2 failures were of pictographs (50%).

The figure below shows the results of the model prediction for each image.  Instances where the prediction did not match the ground truth label ID are highlighted in red.

![alt text][image10]



#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The figure below shows the top 5 softmax probabilities for each of the 30 images found on the web.  The probabilities are shown as a bar chart, which also displays each probability as a percentage.  In most cases, the top prediction probability is at or near 100%.  This is true both when the model as correctly and incorrectly predicted the image.  As a result, the bar chart displays the bar for only the top prediction softmax probability.  In some cases, however, the top 5 softmax probabilities are mixed.  This can be seen more evidently in the 30 kmh zone photograph, where the top softmax probabilty is only 0.1289 (12.89%).

![alt text][image11]



### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

Both max pooling layers were output during training, which could be visualized.  From the visualizations, it appears that the neural network model used detected edges to make classifications.  For example, in the image below, high contrast areas were picked out as features to learn consistent patterns in order to properly classify the image.

![alt text][image12]
