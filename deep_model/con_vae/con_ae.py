from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
import matplotlib.pyplot as plt
import numpy as np
import os

f = np.load('mnist.npz')
train_data = f['x_train'].T #(28, 28, 60000)
trX = train_data.reshape((-1, 28, 28, 1)).astype(np.float32) #(60000, 28, 28, 1)
test_data = f['x_test'].T #(28, 28, 10000)
teX = test_data.reshape((-1, 28, 28, 1)).astype(np.float32) #(10000, 28, 28, 1)
x_train, x_test = trX / 255., teX/255.

input_img = Input(shape=(28,28,1)) # Tensorflow后端， 注意要用channel_last

# 编码器部分
x = Conv2D(16, (3,3), activation='relu', padding='same')(input_img) #28*28*16
x = MaxPooling2D((2,2), padding='same')(x) #14*14*16
x = Conv2D(8,(3,3), activation='relu', padding='same')(x) #14*14*8
x = MaxPooling2D((2,2), padding='same')(x) #7*7*8
x = Conv2D(8, (3,3), activation='relu', padding='same')(x) #7*7*8
encoded = x

# 解码器部分
x = Conv2D(8, (3,3), activation='relu', padding='same')(encoded) #7*7*8
x = UpSampling2D((2, 2))(x) #14*14*8
x = Conv2D(8, (3,3), activation='relu', padding='same')(x) #14*14*8
x = UpSampling2D((2, 2))(x) #28*28*8
x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)#28*28*16
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = Model(inputs=input_img, outputs=decoded)
encoder = Model(inputs=input_img, outputs=encoded)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

# 训练模型， 并且在callbacks中使用tensorBoard实例， 写入训练日志 http://0.0.0.0:6006
from keras.callbacks import TensorBoard
autoencoder.fit(x_train, x_train,
                epochs=200,
                batch_size=128,
                shuffle=True,
                validation_data=(x_test, x_test))
                #callbacks=[TensorBoard(log_dir='/tmp/autoencoder')])

encoded_imgs = encoder.predict(x_test)
np.save("x_test.npy",encoded_imgs)
'''
# 可视化训练结果， 我们打开终端， 使用tensorboard
# tensorboard --logdir=/tmp/autoencoder # 注意这里是打开一个终端， 在终端里运行



# 重建图片

decoded_imgs = autoencoder.predict(x_test)
encoded_imgs = encoder_model.predict(x_test)
n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
    k = i + 1
    # 画原始图片
    ax = plt.subplot(2, n, k)
    plt.imshow(x_test[k].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    # 画重建图片
    ax = plt.subplot(2, n, k + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

# 编码得到的特征
n = 10
plt.figure(figsize=(20, 8))
for i in range(n):
    k = i + 1
    ax = plt.subplot(1, n, k)
    plt.imshow(encoded[k].reshape(4, 4 * 8).T)
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()  
'''              
