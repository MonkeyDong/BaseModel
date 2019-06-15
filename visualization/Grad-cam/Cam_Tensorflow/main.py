from vgg import vgg16
import tensorflow as tf
import numpy as np
from skimage import io
from skimage.transform import resize
#import matplotlib
#matplotlib.use('Agg')
from matplotlib import pyplot as plt
from imagenet_classes import class_names
from scipy.misc import imread, imresize

flags = tf.app.flags
flags.DEFINE_string("input", "laska.png", "Path to input image ['laska.png']")
flags.DEFINE_string("output", "laska_save.png", "Path to input image ['laska_save.png']")
flags.DEFINE_string("layer_name", "pool5", "Layer till which to backpropagate ['pool5']")

FLAGS = flags.FLAGS


def load_image(img_path):
	print("Loading image")
	img = imread(img_path, mode='RGB')      ###img.shape (227, 227, 3)
	img = imresize(img, (224, 224))

	# Converting shape from [224,224,3] to [1,224,224,3]
	x = np.expand_dims(img, axis=0)

	# Converting RGB to BGR for VGG (::-1取反)
	x = x[:,:,:,::-1]      ###x.shape (1, 224, 224, 3)
	return x, img



def grad_cam(x, vgg, sess, predicted_class, layer_name, nb_classes):
	print("Setting gradients to 1 for target class and rest to 0")
	# Conv layer tensor [?,7,7,512]
	conv_layer = vgg.layers[layer_name]
	# layers[pool5]
	# [1000]-D tensor with target class index set to 1 and rest as 0
	
	one_hot = tf.sparse_to_dense(predicted_class, [nb_classes], 1.0)
	#第一个参数sparse_indices：稀疏矩阵中那些个别元素对应的索引值
	#第二个参数output_shape：输出的稀疏矩阵的shape
	#第三个参数sparse_values：个别元素的值。
	#生成一个1000个元素的one_hot向量

	signal = tf.multiply(vgg.layers['fc3'], one_hot)
	#multiply点乘 
	loss = tf.reduce_mean(signal)
	#？？？

	grads = tf.gradients(loss, conv_layer)[0]
	#tf.gradients(ys, xs)实现ys对xs求导
	# Normalizing the gradients
	norm_grads = tf.div(grads, tf.sqrt(tf.reduce_mean(tf.square(grads))) + tf.constant(1e-5))
	#tf.div除法 tf.sqrt平方根 tf.square平方

	output, grads_val = sess.run([conv_layer, norm_grads], feed_dict={vgg.imgs: x})
	output = output[0]           # [7,7,512]
	grads_val = grads_val[0]	 # [7,7,512]

	weights = np.mean(grads_val, axis = (0, 1)) 			# [512]
	cam = np.ones(output.shape[0 : 2], dtype = np.float32)	# [7,7]

	# Taking a weighted average
	for i, w in enumerate(weights):
	    cam += w * output[:, :, i]

	# Passing through ReLU
	cam = np.maximum(cam, 0)
	cam = cam / np.max(cam)
	cam = resize(cam, (224,224))

	# Converting grayscale to 3-D
	cam3 = np.expand_dims(cam, axis=2)#(224,224)->(224,224,1)
	cam3 = np.tile(cam3,[1,1,3]) #np.tile就是把数组沿各个方向复制
	#(224,224,1)->(224,224,3)

	return cam3
	#返回热图？


def main(_):
	x, img = load_image(FLAGS.input)
	###x.shape (1, 224, 224, 3)
	###img.shape (224, 224, 3)

	sess = tf.Session()

	print("\nLoading Vgg")
	imgs = tf.placeholder(tf.float32, [None, 224, 224, 3])
	vgg = vgg16(imgs, 'vgg.npz', sess)

	print("\nFeedforwarding")
	prob = sess.run(vgg.probs, feed_dict={vgg.imgs: x})[0]#probs是网络中fc3输出的结果
	preds = (np.argsort(prob)[::-1])[0:5]
	#argsort函数返回的是数组值从小到大的索引值,[::-1]倒顺序
	print('\nTop 5 classes are')
	for p in preds:
	    print(class_names[p], prob[p])

	# Target class
	predicted_class = preds[0]
	# Target layer for visualization
	layer_name = FLAGS.layer_name
	# Number of output classes of model being used
	nb_classes = 3

	cam3 = grad_cam(x, vgg, sess, predicted_class, layer_name, nb_classes)
	np.save('cam.npy',cam3)

	img = img.astype(float)
	img /= img.max()

	# Superimposing the visualization with the image.
	#将可视化与图像叠加在一起
	new_img = img+3*cam3
	new_img /= new_img.max()
	

	# Display and save
	io.imshow(new_img)
	plt.show()
	io.imsave(FLAGS.output, new_img)
	print('-'*19,'运行完毕','-'*19)


if __name__ == '__main__':
	tf.app.run()
	








