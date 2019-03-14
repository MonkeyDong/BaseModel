from tensorflow.python.keras.applications import ResNet50
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, GlobalAveragePooling2D
#分类类型数
num_classes = 3

resnet_weights_path = 'resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'

my_new_model = Sequential() #Create an empty model
#Add the ResNet-50 without the output layer
my_new_model.add(ResNet50(include_top=False, pooling='avg', weights=resnet_weights_path))
#Add the new output layer
my_new_model.add(Dense(num_classes, activation='softmax'))

# Say not to train first layer (ResNet) model. It is already trained
my_new_model.layers[0].trainable = False

#Use Stochastic gradient descent to optimize, use crossentropy function as loss function
my_new_model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

from tensorflow.python.keras.applications.resnet50 import preprocess_input
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

image_size = 224
data_generator = ImageDataGenerator(preprocessing_function=preprocess_input)

#假如We have 72 picture, so we make the batch size to 12, epoch for 6 steps 
#'data/train'是训练集地址
#'data/validation'是测试集地址
train_generator = data_generator.flow_from_directory(
        'data/train',
        target_size=(image_size, image_size),
        batch_size=12,
        class_mode='categorical')

validation_generator = data_generator.flow_from_directory(
        'data/validation',
        target_size=(image_size, image_size),
        class_mode='categorical')

my_new_model.fit_generator(
        train_generator,
        steps_per_epoch=10,
        epochs=100,
        validation_data=validation_generator,
        validation_steps=1)

