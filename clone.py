import csv
import cv2
import numpy as np
import sklearn


from keras.models import Sequential, Model
from keras.layers import Flatten, Dense,Dropout
from keras.layers import Lambda,Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D


lines =[]
with open('./data2/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
		

images, measurements = [], []
for line in lines:
    for i in range(3):
        source_path=line[i]
        #filename=source_path.split('\\')[-1]
		filename = source_path.split('/')[-1]
        current_path='./data2/IMG/'+filename
        img = cv2.imread(current_path)
        #img1 = cv2.resize(img,(200,66), interpolation = cv2.INTER_AREA)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        augment = .25+np.random.uniform(low=0.0, high=1.0)
        img[:,:,2] = img[:,:,2]*augment
        image = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
        
        images.append(image)
    measurement = np.float64(line[3])
    correction=0.2
    measurements.append(measurement)
    measurements.append(measurement+correction)
    measurements.append(measurement-correction)
# flipping images to augment data set 
augmented_images, augmented_measurements = [], []
for image,measurement in zip(images, measurements):
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    augmented_images.append(cv2.flip(image,1))
    augmented_measurements.append(measurement*-1.0)

from sklearn.model_selection import train_test_split
train_samples, validation_samples, angles_train, angles_validation = train_test_split(augmented_images, augmented_measurements,test_size=0.2, random_state=42)

#generator 
def generator(samples, tr_or_val ,batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            if tr_or_val==0:
                batch_angles = angles_train[offset:offset+batch_size]
            elif tr_or_val==1:
                batch_angles = angles_validation[offset:offset+batch_size]
            else:
                print("Error input.")  
            X_train=np.array(batch_samples)
            y_train=np.array(batch_angles)
            
               
            yield sklearn.utils.shuffle(X_train, y_train)
			
### compile and train the model using the generator function
### version #2
train_generator = generator(train_samples, 0,batch_size=32)
validation_generator = generator(validation_samples,1,batch_size=32)

###
model = Sequential()
# Preprocess incoming data, centered around zero with small standard deviation 
model.add(Lambda(lambda x: (x / 127.5) - 1, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))
model.add(Convolution2D(24,5,5,subsample=(2,2),border_mode='same',activation="relu"))
model.add(Convolution2D(36,5,5,subsample=(2,2),border_mode='same',activation="relu"))
model.add(Convolution2D(48,5,5,subsample=(2,2),border_mode='same',activation="relu"))
model.add(Convolution2D(64,3,3,subsample=(2,2),border_mode='same',activation="relu"))
model.add(Convolution2D(64,3,3,subsample=(2,2),border_mode='same',activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dropout(0.2))
model.add(Dense(50))
model.add(Dropout(0.3))
model.add(Dense(10))
model.add(Dropout(0.3))
model.add(Dense(1))
#model.summary()

from keras.models import Model
import matplotlib.pyplot as plt

### mse  adam optimizer
model.compile(loss='mse', optimizer='adam')
history_object = model.fit_generator(train_generator, samples_per_epoch =  len(train_samples), 
			validation_data=validation_generator,nb_val_samples= len(validation_samples), nb_epoch=6)
model.save('model.h5')
### print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()