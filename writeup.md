# **Behavioral Cloning** 

## Writeup Template

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./images/center.jpg "Center camera"
[image3]: ./images/flip1.jpg "Flip image"
[image4]: ./images/flip2.jpg "Flip image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 5x5 to 3x3 filter sizes and depths between 24 and 64 (model.py lines 77-89). It is based on End-to-end nVidia design. 

Data is normalized in the model using a Keras lambda layer (code line 75). Convolutional layers contain Relu activation function.

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 91 and 98). 

The model was trained and validated on different data sets to ensure that the model was not overfitting, generator always returns different dataset . The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road, driving vehicle on different surfaces and left and right curves.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

My first step was to use a convolution neural network model similar to the LeNet design I thought this model might be appropriate because it consists of 2 convolutional networks which could indetify shape of the road based on lines and edges find on images.

In order to gauge how well the model was working, I created a generator of images and steering data which returns different dataset. Validation and training dataset are based on same source of images but dataset are different.

To prevent underfitting I created simple augunmentation which adds to dataset all images flipped with steering angle inverted.

The result was not good, shape of the road was not indentified and vehicle went out of the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 77-89) was inspisred by End-to-end Nvidia design. It consists of 5 convolutional layers and 5 fully connected layers.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track. Problem was with 2 curves between bridge and starting point. To improve the driving behavior in these cases, I took special trainig for this 2 cases.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I used images taken from cameras placed on sides. It helps to prevent underfitting. I set correction parameter, after few tries, to 0.23. 

To augment the data set, I also flipped images and angles thinking that this would prevent underfitting and speeds up training because all curves are trained for both directions.

Recording of driving track the oposite direction helped as well.

After the collection process, I had 48 663 number of data points. I then preprocessed this data by changing size of images by 2 and cropped images to use for training only usable part of the image.

Example of preprocessed and augumented images:

![alt text][image3] ![alt text][image4]

I finally randomly shuffled the data set and put 33% of the data into a validation set. I set batch size to 120.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5, loss started to grow for more epochs and result was not increased a lot. I used an adam optimizer so that manually training the learning rate wasn't necessary.
