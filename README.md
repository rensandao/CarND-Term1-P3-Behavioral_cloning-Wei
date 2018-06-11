# **Behavioral Cloning Project**

---
This project is aiming to utilize deep learning technology to help car drive safely in automonous mode, in the simulator offered by Udacity.
The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* `model.py` containing the script to create and train the model
* `drive.py` for driving the car in autonomous mode
* `model.h5` containing a trained convolution neural network 
* `README.md` summarizing the results
* `run_1.mp4` recording car's running for one lap in autonomous mode on track one 

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 2x2 filter sizes and depths between 24 and 64 (model.py lines 83-87) 

The model includes activation "relu" to introduce nonlinearity (code line 83-87), and the data is normalized in the model using a Keras lambda layer (code line 81 ). A cropping method is after Lambda layer(code line 82)

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 90,92,94). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 102).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. The data set for track one I finally used was offered by [Project Resources](https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584f6edd_data/data.zip), which contained images from three cameras. Here is the Data Map for `track one` and `track two`.

Track 1

<img src="./Visualized Images/Track1-Data Map.png" width="400px"> <img src="./Visualized Images/Track1-Steering Angle Map.png" width="400px">

Track 2

<img src="./Visualized Images/Track2-Data Map.png" width="400px"> <img src="./Visualized Images/Track2-Steering Angle Map.png" width="400px">

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

`The overall strategy` for deriving a model architecture was to try different neural networks, fed them with a certain amount of picture data, and train for minimum loss and valitdation loss. And the last key step was to check the car driving in autonomous mode, using the deep learning results. 

My first step was to use a convolution neural network model similar to the LeNet-5. I thought this model might be appropriate because I had constructed and applied it in Traffic-Sign-Classifier project, which proved its strong learning ability and effect.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. The first data set I used was collected from the simulator, I ran the car just for over one lap, keeping it ran smoothly in the middle of the lane, but taking several adjustments on purpose from the lane edge to the center of lane. It contained pictures collecting from three cameras mounted in center, left and right position in car.

For the first try, I only took the images collecting from the center camera to fit the LeNet5 network. In 10 epochs, The validation loss is higher than MSE loss, it implied the model was overfiting and the validation was getting higher from the 4th epoch. In real simulation, the car drove badly and got out of the lane from left side.  

To solve the offset, I augmented the data set by flipping the original images, it worked better in distance, but got out of the lane in right side. Then I included left and right camera images with the steering angles adjusted. Thus, the data set amount inproved by 4 times. This time, 
It succeeded to run across two turning lane, but failed in third turn corner, which lacks obvious lane line.

I refered that sky, trees and the car hood in images made bad effect on deep learning. So I included Cropping2D function as recommended in course video to cut them and It did well and the car sucessfully ran across the corner. 

I found that several adjustment above,  the validation set loss increased  every time after the 4th epoch and was also higher than the training set loss. This implied that the model's overfitting stayed still. To combat the overfitting, I modified the model by adding three dropout layers, it went better, but cannot perfectly remove.

Then I changed to a new stronger network, which had been proved by NVIDIA Team-[nVidia model](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf). 

Here is their basic CNN architecture.

<img src="./Visualized Images/nVidia_model.png" width="400px">

It did well in training, whose results implied the loss on training set and validation set were both low. But overfitting also accurred. So I also modified the model to get rid of overfitting by incluing Dropout function and color channel changing function.  At the end of the process, the car was able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture is based on Nvidia mature structure(model.py lines 83-95), but included some changes, such as adding dropout layers to minimize overfitting.

Here is a visualization of the architecture 

<img src="./Visualized Images/Architecture visualization.png" width="500px" height="500px">

#### 3. Creation of the Training Set & Training Process

For Track 1, I abandoned my collecting data, and used the data offered by `course material`. To capture good driving behavior, It mainly recorded two laps using center lane driving. Here is an example image of center lane driving:

<img src="./Visualized Images/Track1_center0.jpg" width="200px">

It also recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to get back to center. These images show what a recovery looks like starting from right :

<img src="./Visualized Images/Track1_center1.jpg" width="150px">   <img src="./Visualized Images/Track1_center2.jpg" width="150px">  <img src="./Visualized Images/Track1_center3.jpg" width="150px">

And It recorded the reverse running data, which can improve the nomalization. On track 2, I repeated this process to get data points.

To augment the data set, I also flipped images and angles thinking that this would helply adjust steering angles. For example, here are images from three cameras that has then been flipped:

Track 1

<img src="./Visualized Images/Track1-flipping.png" width="400px">

Track 2

<img src="./Visualized Images/Track2-flipping.png" width="400px">



#### 4. Generator

The amount of images in the data set after augmenting became 6 times larger. It was so large that my memory was occupied by 98 at most, and got stuck. So I have to choose the AWS fly sevice, which really made effect.

But to train these data,  it was also a big tough to load all the data for only one time. So it's better to use patchs to feed to network, which is shown in the last project-Traffic Signs Classifier. Here it is `generator` function method, which can realize it very  well.

There are two ways to use `generator` in this case, with small difference:

(1) Imread the data in advance, and fliped or other augmented ways to deal with images. Then split them to train and validation samples. Finally, feed every patch data to network model.

(2) Unlike imreading before generator, split the csv file (mainly the images' paths and steering angles) firstly, Then in the generation function, imread and augmented a patch of data every batch size.

#### 5. Test results

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5 after trials.

Here is a visualization of the Loss Map:

<img src="./Visualized Images/Track1-loss.png" width="400px">

Here is the final running [video](./run_1.mp4) on track one. Video for track two is currently on the road. 

### Conclusion and Disscussion

Firstly, Keras shows its great convenience in fomat using compared to TensorFlow. In TensorFlow, it need detailed arguments to be added
one by one, easily confused, but Keras offered better interface which got things done easily in personal idea.

Secondly, for Behavioral Cloning. Data collecting work is hard but definitely important and patience-needed. The quality of data set matters even more than architecture in some degree. So how to collect different effective and viable data should awalys be considered as good way. So it need some technique and more patience that personally need to enhance.

besides, data augment techniques need to add based on different scene, such as track one and track two. Currently I haven't realize it , cause I need more time.

Finally, with data augmented and strong CNN architecture trained, car can basically lead its own way on track. But there are a lot work to do since we only consider the steering angle. It is so complicated that we have further way to go.

