import keras 
from keras.preprocessing.image import ImageDataGenerator 
from keras.models import Sequential 
from keras.layers import Dense 
from keras.layers import Dropout 
from keras.layers import Flatten 
from keras.optimizers import SGD 
from keras.layers.convolutional import Conv2D 
from keras.layers.convolutional import MaxPooling2D 
from keras.utils import np_utils 
from keras.layers import LeakyReLU

n = 168
ratio = 0.2
batch_size = 1
 
train_datagen = ImageDataGenerator(rescale=1/255., 
shear_range=0.2, 
zoom_range=0.2, 
horizontal_flip=True 
) 
val_datagen = ImageDataGenerator(rescale=1/255.) 

train_generator = train_datagen.flow_from_directory( 
'data/train/', 
target_size=(150, 150), 
batch_size=batch_size, 
class_mode='categorical') 
 
validation_generator = val_datagen.flow_from_directory( 
'data/validation/', 
target_size=(150, 150), 
batch_size=batch_size, 
class_mode='categorical')

model = Sequential() 
 
model.add(Conv2D(32, (3, 3), input_shape=(150, 150, 3), padding='same'))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D(pool_size=(2, 2))) 

model.add(Conv2D(32, (3, 3), padding='same'))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D(pool_size=(2, 2))) 

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D(pool_size=(2, 2))) 

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D(pool_size=(2, 2))) 
 
model.add(Conv2D(128, (3, 3), padding='same'))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D(pool_size=(2, 2))) 

model.add(Conv2D(256, (3, 3), padding='same'))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D(pool_size=(2, 2))) 

model.add(Dropout(0.25)) 
model.add(Flatten()) 
model.add(Dense(1024))
model.add(LeakyReLU(alpha=0.1))
model.add(Dropout(0.5)) 
model.add(Dense(64)) 
model.add(LeakyReLU(alpha=0.1))
model.add(Dropout(0.5)) 
model.add(Dense(2, activation='softmax')) 

epochs = 50 
lrate = 0.001
decay = lrate/epochs 
sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False) 
model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])


fitted_model = model.fit_generator( 
train_generator, 
steps_per_epoch= int(n * (1-ratio)) // batch_size, 
epochs=100, 
validation_data=validation_generator, 
validation_steps= int(n * ratio) // batch_size)

























