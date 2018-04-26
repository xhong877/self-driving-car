import csv
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import random

lines = []
headline = True
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        if headline == True:
            headline = False
        else:
            lines.append(line)

from sklearn.cross_validation import train_test_split
train_samples, validation_samples = train_test_split(lines, test_size=0.2)


def generator(samples,batch_size=32):
    num_samples = len(samples)
    while 1:
        random.shuffle(samples)
        for offset in range(0,num_samples,batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            measurements = []
            augmented_images,augmented_measurements=[],[]
            for batch_sample in batch_samples:
                for i in range(3):
                    source_path = batch_sample[i]
                    filename = source_path.split('/')[-1]
                    current_path = './data/IMG/' + filename
                    image = plt.imread(current_path)
                    images.append(image)
                    if i == 0:
                        measurement = float(line[3])
                    elif i == 1:
                        measurement = float(line[3])+0.2
                    else:
                        measurement = float(line[3])-0.2
                        measurements.append(measurement)

                for image,measurement in zip(images,measurements):
                    augmented_images.append(image)
                    augmented_measurements.append(measurement)
                    augmented_images.append(np.fliplr(image))
                    augmented_measurements.append(-1.0*measurement)

            X_train = np.array(augmented_images)
            y_train = np.array(augmented_measurements)
            yield sklearn.utils.shuffle(X_train, y_train)

train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

from keras.models import Sequential
from keras.layers import Flatten,Dense,Dropout,Cropping2D
from keras.layers import Lambda
from keras.layers.convolutional import Conv2D,MaxPooling2D

model = Sequential()
model.add(Lambda(lambda x:(x/255.0)-0.5,input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))
model.add(Conv2D(96,11,11,subsample = (4,4),border_mode='valid',activation='relu'))
model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))
model.add(Conv2D(256,5,5,subsample=(1,1),border_mode='same',activation='relu'))
model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))
model.add(Conv2D(384,3,3,subsample=(1,1),border_mode='same',activation='relu'))
model.add(Conv2D(384,3,3,subsample=(1,1),border_mode='same',activation='relu'))
model.add(Conv2D(256,3,3,subsample=(1,1),border_mode='same',activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
model.add(Flatten())
model.add(Dense(4096,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(4096,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
performance = model.fit_generator(train_generator, samples_per_epoch= len(train_samples)*6.0, validation_data=validation_generator, nb_val_samples=len(validation_samples)*6.0, nb_epoch=7)
model.save('model.h5')
print(performance.history.keys())
print('Loss')
print(performance.history['loss'])
print('Validation Loss')
print(performance.history['val_loss'])

plt.plot(performance.history['loss'])
plt.plot(performance.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.savefig("history.jpg")  