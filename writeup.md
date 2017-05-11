# Traffic Sign Recognition


**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set on german traffic signs as provided by udacity
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./test_images/0.jpg "Traffic Sign 1"
[image2]: ./test_images/10.jpg "Traffic Sign 2"
[image3]: ./test_images/12.jpg "Traffic Sign 3"
[image4]: ./test_images/13.jpg "Traffic Sign 4"
[image5]: ./test_images/14.jpg "Traffic Sign 5"
[image6]: ./test_images/3.jpg "Traffic Sign 6"
[image7]: ./test_images/33.jpg "Traffic Sign 7"
[image8]: ./test_images/35.jpg "Traffic Sign 8"

[histo]: ./histo.png
[traffic_sign_samples]: ./traffic_sign_samples.png
[test_image_results]: ./test_image_results.png
[featuremap]: ./featuremap.png

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

### Data Set Summary & Exploration

* There are 34799 images in the training set.
* There are 4410 images in the validation set.
* There are 12630 images in the test set.
* The shape of a traffic sign image is 32x32 RGB image.
* The number of unique classes/labels in the data set is 43.

Here is a histogram on how frequently each traffic sign appears in the training set. 

![alt text][histo]

The above histogram shows that some signs (around 15 signs) appear at much more often than other signs. The less frequent signs may be difficult to train in some cases.

Here is a plot to display 8 random sample images from each traffic sign class. This helps understand how the actual images from traing set look. Many such samples have low brightness. Some images are distorted when some other images are hard to recognize.

![alt text][traffic_sign_samples]

### Design and Test a Model Architecture

#### 1. Data preprocessing
As the input datasets are already sized for 32x32 RGB images, no further image resizing is done. Each of color channels have int8 values (0-255) which need to be normalized before being fed to a neural network. So, each image in training, validation and test datasets is normalized via a function, f(x) = (x - 127.5) / 127.5 to restrict each color channel in [-1.0, 1.0] range. This input range is suitable for training a typical neural network and also allows a network to use simpler relu activations to introduce non-linearity. Finally, training dataset is shuffled to break any pattern sequence in the dataset.

#### 2. Datasets
There are three datasets given in this project - training, validation and test datasets. Training datasets are used to train a neural network model. Validation dataset is used to cross-validate the trained model at every epoch of training. Finally test dataset is tested using the final trained model to check accuracy against ground truth.

#### 3. Model architecture

The final model is basically a LeNet model with the addition of dropout layers for hidden fully connected layers. It consists of below layers.

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| Relu					|												|
| Max pooling 2x2     	| 2x2 stride, valid padding, outputs 14x14x6    |
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16   |
| Relu					|												|
| Max pooling 2x2     	| 2x2 stride, valid padding, outputs 5x5x16     |
| Flatten		        | outputs 400x1        							|
| Fully-connected	    | outputs 120x1     							|
| Relu					|												|
| Dropout				| 												|
| Fully-connected	    | outputs 84x1     			     				|
| Relu					|												|
| Dropout				| 												|
| Fully-connected	    | outputs 43x1     			     				|
| Softmax				| outputs 43x1									|

#### 4. Training the model

The above model is trained on the training dataset using cross-entropy loss function with regularization and with the help of an Adam optimizer which does not need a learning rate. The training is done for 40 epochs and for each epoch, a batch of 128 training samples is utilized at a time to train the model. The regularization parameter (lambda) is chosen 0.001 and probability of keeping neurons active in dropout layers is 0.5.

#### 5. Finding a solution

A basic LeNet model architecture is chosen first to train a traffic sign classifier in this project. LeNet model was originally developed to recognize hand-written digits in images. Similarly, a traffic sign classifier needs to recognize certain shapes in the images and to classify them into different labels. This is why LeNet model is chosen as a baseline model for recognizing traffic signs. 

However, a basic LeNet model does not produce as high validation accuracy as recommended in this project. It is likely that vanilla LeNet model underfits the dataset. So, the LeNet model is augmented to multiple dropout layers in hidden and fully connected layers with the expectation that randomly dropped neurons in these fully connected layers can learn extra features from their inputs. After this adjustment, modified LeNet model improves validation accuracy beyond the requirement.

The final model produces validation accuracy of 95.8% and test accuracy of 94.3%. 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are 8 German traffic signs that I found on the web:

![alt text][image1]
![alt text][image2]
![alt text][image3] 
![alt text][image4]
![alt text][image5]
![alt text][image6]
![alt text][image7]
![alt text][image8]

A couple of above images are not just traffic signs but they have other unrelated parts exposed in their bounding boxes. These could be difficult to classify.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set.

The model predictions for the above test images are the following.

| Image			                                |     Prediction	        					| 
|:---------------------------------------------:|:---------------------------------------------:| 
| Speed limit (20km/h)      		            | Speed limit (20km/h)   					    | 
| No passing for vehicles over 3.5 metric tons  | No passing for vehicles over 3.5 metric tons  |
| Priority road					                | Priority road								    |
| Yield					                        | Yield											|
| Stop					                        | Stop											|
| Speed limit (60km/h)	      		            | Vehicles over 3.5 metric tons prohibited	    |
| Turn right ahead		    	                | Turn right ahead      					    |
| Ahead only			                        | Ahead only      							    |

In details, below are bar charts for softmax probabilities for top 5 traffic signs as predicted by the model for each new test image.

![alt text][test_image_results]

The "Speed limit (60km/h)" sign is misclassified as "Vehicles over 3.5 metric tons prohibited". The model appears to have found red circle that is common to these two signs but could not finally distinguish the digits inside.
The "Speed limit (20km/h) " sign is correctly predicted by the model. The model assigns this correct label a softmax probability of 0.65 and the second label ("Speed limit (20km/h) ") a probability of 0.15. Since these two signs are close enough (diffing by a digit), this is expected.
The other traffic signs are correctly classified by the model with very high softmax probability.

The model is able to correctly guess 7 of the 8 traffic signs, which gives an accuracy of 87.5%. This is less than 94.3% accuracy observed in test dataset. Since this new testset hand-picked from web is very small, its accuracy can be lower than observed accuracy in a much larger testset.
