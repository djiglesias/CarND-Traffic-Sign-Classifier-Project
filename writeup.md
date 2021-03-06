# **Traffic Sign Recognition** 

This project explores how to use TensorFlow to train a neural network to recognize German traffic signs. A data set provided by Udacity contains a large sample set of 32x32 images and their corresponding classifer.

---
## 1. Data Set Summary & Exploration

### 1.1 Basic summary of the data set.

* The size of training set is 34799.
* The size of the validation set is 4410.
* The size of test set is 12630.
* The shape of a traffic sign image is 32x32x3.
* The number of unique classes/labels in the data set is 43.

### 1.2 Visualization of the data set.

The original data set contained an uneven distribution of images amoungst the traffic sign classes with the minimum  of 180 and a maximum of 2010 images in a single class. The histogram below shows the frequency of images in each of the data sets (training, validation, testing) with the frequency of on the y-axis and sign classification on the x-axis.

<p align="center">
  <img src="./images/histogram_raw.png">
</p>

---
## 2. Design & Test a Model Architecture

### 2.1 Preprocessing the Data Set
Functions for manipulating the image data sets are contained within the SelectionTools() class and include options for applying actions such as grayscale, rotation, brightness adjustments, normalization, and data augmentation to generate fake data to even out the class frequencies.

#### 2.1.1 Grayscaling the Images
The images were converted to grayscale reducing their size from 32x32x3 to 32x32x1 which ultimately reduces the overhead for the neural network by a third. Grayscale allows an image to be represented as shades and can better be converted into gradients than if using an RGB image. Since TensorFlow is looking for a variable of size (?, 32, 32, 1) to train the network, I experienced issues with image size when converting to scale with OpenCV since it deemed the fourth dimension of '1' unnecessary and returned an image set of size (?, 32, 32) which is incompatible. Instead, I used numpy.sum() with option `keepdims=True` to average the three colour channels to a single value while maintaining a compatible shape. 

<p align="center">
  <img src="./images/grayscale.png">
</p>

#### 2.1.2 Normalize the Images
The grayscaled images were normalized from range (0,255) to (-1,1) to reduce computation when training the network.

<p align="center">
  <img src="./images/normalize.png">
</p>

#### 2.1.3 Augment Training Data Set
Since the data set was unevenly distributed, my worry was that the network would train more on certain signs more than others. To even the playing field I simply concatenated the images of each class to themselves until they reached a minimum threshold provided (4000 in my case) so that all the classes would have an equal prediction accuracy.

<p align="center">
  <img src="./images/histogram_post.png">
</p>

### 2.2 Model Architecture
  
| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscale image   							| 
| Convolution 5x5    	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Convolution 5x5    	| 1x1 stride, valid padding, outputs 14x14x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				|
| Convolution 5x5    	| 1x1 stride, valid padding, outputs 1x1x400 	|
| RELU					|												|
| Flatten          | outputs 400  |
| Concatenate      | outputs 800  |
| Dropout          |   | 
| Fully connected		| outputs 43					|

---
## 3. Training Model
To train the model several steps were performed to randomize the data before each epoch that included shuffling, rotating, and altering the brightness randomly. Since a large amount of data was augmented there will be many repeated images in the data set, so by randomizing the data it will help differientiate the data set to create unique images.

### 3.1 Shuffle the Data
To prevent the model from memorizing the order of images, the data is shuffled at random every time while maintaining the alignment between the image and the respective label. The master data set is left untouched as to prevent irreversible damage and a copy is made each time on which the SelectionTool() actions are performed on.

### 3.2 Random Rotation
All the images are randomly rotated 90&deg left or right, or left the same. Experimentation with rotating at angles between 0-90 was ommited as the resulting image left a black border around the perimeter that seemed to through false classifications from gradients created by artifacts.

<p align="center">
  <img src="./images/rotate.png" />
</p>

### 3.3 Random Brightness
Each image in the data set is adjusted for brightness, either increasing or decreasing the average value of the pixels each time. Again, it is import to note that the original data set is copied each time to avoid the case where an image is continually increased in brightness until fully saturated and the image data is lost.

<p align="center">
  <img src="./images/brightness.png" />
</p>

### 3.4 Model Parameters
The training parameters used did not differ much from the LeNet lab, with exception to a slightly lower learning rate and several more epochs to account for the lower rate.


| **Parameter**        		|     **Value**	  	| 
|:---------------------:|:-----------------:| 
| Sigma | 0.1 |
| Mu | 0 |
| Batch Size | 128 |
| Epochs | 30 |
| Learning Rate | 0.0008 |

---
## 4. Approach for Training Model
The final model results were a validation set accuracy of 95.2% and a test set accuracy of 93.0%. Initially, the model architecture from the lab was chosen to train the model as demonstrated in the course project notes however this deemed unsatisfactory for the reaching the necessary model accuracy. In order to increase the model accuracy, steps were added as suggested by the course lecture material.

### 4.1 Adding Dropout
A dropout layer was added at the end of layer 4 and was set to 50.0% to set half of the weights to zero forcing the model to learn redundant methods for classifying images. However, for the evaluation of the model against the validation and testing data sets the dropout value was restored to 100% so that no values were dropped.

### 4.2 Modifying Shape
The model was kept at an input shape of (?, 32, 32, 1) rather than switching to (?, 32, 32, 3) since the data being feed into the model was grayscale and not RGB images. However, the model is dynamic so it can handle grayscale or rgb images regardless.

### 4.3 Tuning Hyperparameters
To increase the model accuracy a series of iterations of the training rate was attempted starting by reducing it by half to 0.0005, but after several runs the final value was set to 0.0008. Reducing the training rate required more than 10 epochs fully train, therefore the model was increased to 50 epochs but after watching the model train it was decided that 30 epochs was sufficient. The hyperparameters mu, sigma, and batch size were unaltered from the original model. 

<p align="center">
  <img src="./images/error.png" />
</p>

---
## 5. Test a Model on New Images

### 5.1 German Traffic Signs from the Web

For this project six German traffic signs were found on the web and used to test the accuracy of the model on images not previously viewed by the model. The majority of the signs contain features that are easy to classify by the model as they are simple geometric shapes with large edges (either straight or round) with exception one which contains multiple fine details and proves difficult for the model to classify.


| Image			                   |     Class	    | Pros | Cons |
|:----------------------------:|:-------------:|:-------------------------------:|:-----------------------:|
| <img src="./images/18.png"/>  | 18 | Large simple shapes, triangular sign with thick border, no small details on actual sign |  Messy background, cropped sign below |
| <img src="./images/1.png"/>   | 1  | Large simple shapes, circular sign with thick border, no small details on actual sign | Messy background, large shapes behind image |
| <img src="./images/3.png"/>   | 3  | Large simple shapes, circular sign with thick border, no small details on actual sign, clear background | Cropped sign above |
| <img src="./images/38.png"/>  | 38 | Large simple shapes, circular sign with thick border, no small details on actual sign | Minor noise in bottom left side of corner  |
| <img src="./images/34.png"/>  | 34 | Large simple shapes, circular sign with thick border, no small details on actual sign | Minor background noise |
| <img src="./images/25.png"/>  | 25 | Large simple triangular sign with thick border, clear background | Small details on actual sign, post running vertically behind sign |

<p align="center">
  <img src="./images/custom_images.png" />
</p>


### 5.2 Model Predictions

| Image			        |     Prediction	       | 
|:-----------------:|:----------------------:| 
| General Caution		| General Caution		  	 | 
| 30 km/h      			| 30 km/h								 |
| 60 km/h	  				| 60 km/h								 | 
| Keep Right	   		| Keep Right	  	 			 |
| Left Turn Ahead		| Left Turn Ahead				 |
| Road Work         | Dangerous Curve Right  |


The model was able to correctly guess 5 of the 6 traffic signs, which gives an accuracy of 83.3%. The five images that were classified correctly contained simple geometric shapes, such as arrows and numbers, with relatively large sides for the model to pick up on while the sign that was misclassified had a more complex shape which was distorted by the low photo resolution. Overall, the model predicted the signs very well.


### 5.3 Model Accuracy
To provide some insight to the model predictions, the below image shows the input image on the left hand side followed by the top five model predictions with their confidence level in percent. For the first five images it can be seen that the model predicted the sign classification correctly with 100.0% confidence while showing 0.0% for the remaining images, however the last image was misclassified and showed a 99.0% confidence. 

<p align="center">
  <img src="./images/prediction.png" />
</p>
