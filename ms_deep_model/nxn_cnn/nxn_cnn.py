from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation,Dropout,Convolution2D,MaxPooling2D,Flatten
import numpy as np

x_train = np.load("save_npy/pic_train.npy")
x_test = np.load("save_npy/pic_test.npy")

y_train = np.load("save_npy/labels_train.npy")
y_test = np.load("save_npy/labels_test.npy")


x_train = x_train.reshape(x_train.shape[0],48,48,1)
x_test = x_test.reshape(x_test.shape[0],48,48,1)

# 将X_train, X_test的数据格式转为float32
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')


# 将类别向量(从0到nb_classes的整数向量)映射为二值类别矩阵，
# 相当于将向量用one-hot重新编码
y_train = np_utils.to_categorical(y_train, num_classes=3)
y_test = np_utils.to_categorical(y_test, num_classes=3)


#################modeling#######################
# 建立序贯模型
model = Sequential()                                           #28*28*1


model.add(Convolution2D(                                       #48*48*25
    filters=25,
    kernel_size=(3,3),
    padding='same',
    input_shape=(48,48,1))) #通道数在后

model.add(MaxPooling2D(                                       #24*24*25
    pool_size=(2,2),
    strides=2))  #默认strides值为pool_size

model.add(Convolution2D(                                       #24*24*50
    filters=50,
    kernel_size=(3,3),
    padding='same')) #通道数在后

model.add(MaxPooling2D(                                       #12*12*50
    pool_size=(2,2),
    strides=2))  #默认strides值为pool_size


model.add(Convolution2D(                                      #12*12*100
    filters=100,
    kernel_size=(3,3),
    padding='same'))


model.add(MaxPooling2D(pool_size=(2,2)))                      #6*6*100


# Flatten层，把多维输入进行一维化，常用在卷积层到全连接层的过渡
model.add(Flatten())                                           #3600

model.add(Dense(units=1024))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(units=128))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(units=3))
model.add(Activation('softmax'))

# 输出模型的参数信息
model.summary()

#######################cconfiguration############
# 配置模型的学习过程
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

#######################training###################
model.fit(x_train,y_train,batch_size=100,epochs=10)

#######################evaluate###################
score=model.evaluate(x_test,y_test)
print('Test accuracy:', score[1])
