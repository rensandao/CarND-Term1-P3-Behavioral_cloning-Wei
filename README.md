# **Behavioral Cloning Project**


---
This project is aiming to utilize deep learning technology to help car drive safely in automonous mode, in the simulator offered by Udacity.
The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results
* video.mp4 recording car's running for one lap in autonomous mode 

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 2x2 filter sizes and depths between 24 and 64 (model.py lines 82-87) 

The model includes activation "relu" to introduce nonlinearity (code line 82-87), and the data is normalized in the model using a Keras lambda layer (code line 81 ). A cropping method is after Lambda layer(code line 82)

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 90,92,94). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 102).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. The data set I finally used was offered by [Project Resources](https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584f6edd_data/data.zip), which contained images from three cameras. Here are the Data Map.
![alt text][image1]
![alt text][image1]

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to try different neural networks, fed them with a certain amount of picture data, and train for minimum loss and valitdation loss. And the last key step was to check the car driving in autonomous mode, using the deep learning results. 

My first step was to use a convolution neural network model similar to the LeNet-5. I thought this model might be appropriate because I had constructed and applied it in Traffic-Sign-Classifier project, which proved its strong learning ability and effect.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. The first data set I used was collected from the simulator, I ran the car just for over one lap, keeping it ran smoothly in the middle of the lane, but taking several adjustments on purpose from the lane edge to the center of lane. It contained pictures collecting from three cameras mounted in center, left and right position in car.

For the first try, I only took the images collecting from the center camera to fit the LeNet5 network. In 10 epochs, The validation loss is higher than MSE loss, it implied the model was overfiting and the validation was getting higher from the 4th epoch. In real simulation, the car drove badly and got out of the lane from left side.  

To solve the offset, I augmented the data set by flipping the original images, it worked better in distance, but got out of the lane in right side. Then I included left and right camera images with the steering angles adjusted. Thus, the data set amount inproved by 4 times. This time, 
It succeeded to run across two turning lane, but failed in third turn corner, which lacks obvious lane line.

I refered that sky, trees and the car hood in images made bad effect on deep learning. So I included Cropping2D function as recommended in course video to cut them and It did well and the car sucessfully ran across the corner. 

I found that several adjustment above,  the validation set loss increased  every time after the 4th epoch and was also higher than the training set loss. This implied that the model's overfitting stayed still. To combat the overfitting, I modified the model by adding three dropout layers, it went better, but cannot perfectly remove.

Then I changed to a new stronger network, which had been proved by `Nvidia` company. It did well in training, whose results implied the loss on training set and validation set were both low. But overfitting also accurred. So I also modified the model to get rid of overfitting by incluing Dropout function and color channel changing function.  At the end of the process, the car was able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture is based on Nvidia mature structure(model.py lines 83-95), but included some changes, such as adding dropout layers to minimize overfitting.

Here is a visualization of the architecture 

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

For Track 1, I abandoned my collecting data, and used the data offered by ```the Course material ```. To capture good driving behavior,It mainly recorded two laps using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

It also recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to get back to center. These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

And It recorded the reverse running data, which can improve the nomalization.

To augment the data set, I also flipped images and angles thinking that this would helply adjust steering angles. For example, here are images from three cameras that has then been flipped:

![alt text][image6]
![alt text][image7]


After the collection process, I had X number of data points. I then preprocessed this data by ...


Of cource, on track 2, I repeated this process to get data points.

#### 4. Generator
The amount of images in the data set after augmenting became 6 times larger. It was so large that my memory was occupied by 98 at most, and got stuck. So I have to choose the AWS fly sevice, which really made effect.

But to train these data,  it was also a big tough to load all the data for only one time. So it's better to use patchs to feed to network, which is shown in the last project-Traffic Signs Classifier. Here it is Generator function method, which can realize it.

There are two ways to use Generator in this case, with small difference:

(1) Imread the data in advance, and fliped or other augmented ways to deal with images. Then split them to train and validation samples. Finally, feed every patch data to network model.

(2) Unlike imreading before Generator, split the csv file (mainly the images' paths and steering angles) firstly, Then in the Generation function, imread and augmented a patch of data every batch size.

#### 5. Test results

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5 after trials.

Here is a visualization of the Loss Map:
![alt text][image5]


### Further improvement


