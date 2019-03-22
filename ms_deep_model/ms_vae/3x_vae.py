import numpy as np
np.random.seed(1337)  # for reproducibility
 
from keras.datasets import mnist
from keras.models import Model #泛型模型
from keras.layers import Dense, Input
from keras import optimizers
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
 
x_train = np.load("save_npy/pic_train.npy")
x_train = x_train.reshape((x_train.shape[0], -1))
x_test = np.load("save_npy/pic_test.npy")
x_test = x_test.reshape((x_test.shape[0], -1))

y_test = np.load("save_npy/labels_test.npy")

print(x_train.shape)
print(x_test.shape,y_test.shape)

# 压缩特征维度至2维
encoding_dim = 3
 
# this is our input placeholder
input_img = Input(shape=(2416,))
 
# 编码层
encoded = Dense(512, activation='relu')(input_img)
encoded = Dense(128, activation='relu')(encoded)
encoded = Dense(64, activation='relu')(encoded)
encoded = Dense(10, activation='relu')(encoded)
encoder_output = Dense(encoding_dim)(encoded)
 
# 解码层
decoded = Dense(10, activation='relu')(encoder_output)
decoded = Dense(64, activation='relu')(decoded)
decoded = Dense(128, activation='relu')(decoded)
decoded = Dense(512, activation='relu')(decoded)
decoded = Dense(2416, activation='tanh')(decoded)
 
# 构建自编码模型
autoencoder = Model(inputs=input_img, outputs=decoded)
 
# 构建编码模型
encoder = Model(inputs=input_img, outputs=encoder_output)

adam = optimizers.Adam(lr=0.00001)
# compile autoencoder
autoencoder.compile(optimizer=adam, loss='mse')
 
# training
autoencoder.fit(x_train, x_train, epochs=20, batch_size=256, shuffle=True)
 
# plotting
encoded_imgs = encoder.predict(x_test)
#plt.scatter(encoded_imgs[:, 0], encoded_imgs[:, 1], c=y_test, s=3)
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(encoded_imgs[:, 0], encoded_imgs[:, 1], c=y_test,s=1)

ax.set_zlabel('Z', fontdict={'size': 15, 'color': 'red'})
ax.set_ylabel('Y', fontdict={'size': 15, 'color': 'red'})
ax.set_xlabel('X', fontdict={'size': 15, 'color': 'red'})

plt.show()

