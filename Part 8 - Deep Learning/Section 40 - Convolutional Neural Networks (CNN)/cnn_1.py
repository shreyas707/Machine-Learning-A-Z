# CNN - Convolution -> Max Pooling -> Flattening -> Full Connection

# Part 1 - Building the CNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
import time

# Initialising the CNN
# initialize the cnn as a sequence of layers
classifier = Sequential()

# Step 1 - Convolution
# Convert image into a table of pixel values (feature map)
# 32 feature detectors (usually start from 32), 3x3 dimensions for feature detectors
# input_shape 3 if coloured, 1 if non coloured. 64x64 is the number of pixels
classifier.add(Convolution2D(32, 3, 3, input_shape=(64, 64, 3), activation='relu'))

# Step 2 - Pooling Step (Reducing size of feature maps)
# Size of feature map is divided by 2
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Adding second convolutional layer
# We don't need to add input_shape as this is the second layer and cnn already knows the value
classifier.add(Convolution2D(32, 3, 3, activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Step 3 - Flattening (take all pooled feature maps and put them into single vector)
classifier.add(Flatten())
 
# Step 4 - Full Connection
classifier.add(Dense(output_dim=128, activation='relu'))
classifier.add(Dense(output_dim=1, activation='sigmoid'))

# Compiling the CNN
# optimizer - stochastic gradient descent algorithm
# loss - binary_crossentropy because its a classification problem and the outcome is binary. for more than 2 outcomes, use categorical_crossentropy 
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Part 2 - Fitting the CNN to the images
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

start_time = time.time()

classifier.fit_generator(training_set,
                         samples_per_epoch = 8000,
                         nb_epoch = 25,
                         validation_data = test_set,
                         nb_val_samples = 2000)

print("--- %s seconds ---" % (time.time() - start_time))